
from pathlib import Path
import datetime
import math
import warnings
import json
from collections import OrderedDict
from typing import Generator
import importlib
import copy

import numpy as np
from tqdm import tqdm
import torch
from torch import optim, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LRScheduler
from accelerate import load_checkpoint_in_model
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from collections import deque

from data.data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, test_params_flop, test_train_time, test_gpu_memory
from utils.metrics import metric
from utils.globals import logger, accelerator
from utils.ExpConfigs import ExpConfigs

warnings.filterwarnings('ignore')


class OfflineUncertaintyEstimator(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, hidden_dim=64):
        super(OfflineUncertaintyEstimator, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.register_buffer("error_min", torch.tensor(0.0))
        self.register_buffer("error_max", torch.tensor(1.0))
        
        in_dim_x = self.seq_len * self.enc_in
        in_dim_y = self.pred_len * self.enc_in

        self.mlp_x = nn.Sequential(
            nn.Linear(in_dim_x, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.mlp_y = nn.Sequential(
            nn.Linear(in_dim_y, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )


        fusion_out = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_out),
            nn.ReLU(),
        )
        self.loss_estimation = nn.Sequential(
            nn.Linear(fusion_out, self.enc_in),
            nn.Sigmoid(), 
        )
        self.quality_estimation = nn.Sequential(
            nn.Linear(fusion_out, self.enc_in),
            nn.Sigmoid(),  
        )

    def forward(self, x, pred):
        """
        forward 
        x: [B, seq_len, enc_in]
        pred: [B, pred_len, enc_in]
        """
        batch_size = x.shape[0]

    
        x_flat = x.reshape(batch_size, -1)                  # [B, seq_len * enc_in]
        pred_flat = pred.reshape(batch_size, -1)            # [B, pred_len * enc_in]


        feat_x = self.mlp_x(x_flat)         # [B, hidden]
        feat_y = self.mlp_y(pred_flat)           # [B, hidden]
        feat = torch.cat([feat_x, feat_y], dim=1)  # [B, hidden*2]
        fused = self.fusion(feat)                  # [B, fusion_out]

        unc_per_channel = self.loss_estimation(fused)        # [B, enc_in]
        alpha = self.quality_estimation(fused)               # [B, enc_in]

        alpha_sum = alpha.sum(dim=1).clamp_min(1e-8)         # [B]
        uncertainty_scores = (unc_per_channel * alpha).sum(dim=1) / alpha_sum  # [B]

        return uncertainty_scores


class AdaptiveGDC_Calibration(nn.Module):
    def __init__(self, seq_len, pred_len, n_var, configs, hidden_dim=64, var_wise=True):
        super(AdaptiveGDC_Calibration, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_var = n_var
        self.configs = configs
        
        self.expert_reliable = Calibration(seq_len, pred_len, n_var, hidden_dim, var_wise=var_wise)
        self.expert_unreliable = Calibration(seq_len, pred_len, n_var, hidden_dim, var_wise=var_wise)
        
        self.ema_mean = None
        self.ema_var = None

        self.ema_alpha = getattr(self.configs, '_ema_alpha_alloc', 0.75)
        self.std_k = getattr(self.configs, '_std_k_alloc', 0.25)
        
    def _update_ema_stats(self, uncertainty_scores):
        if isinstance(uncertainty_scores, torch.Tensor):
            u = uncertainty_scores.detach().cpu().flatten()
        else:
            u = torch.as_tensor(uncertainty_scores)
        
        avg = u.mean().item()
        var = u.var(unbiased=False).item()
        
        if self.ema_mean is None:
            self.ema_mean = avg
            self.ema_var = var
        else:
            a = self.ema_alpha
            self.ema_mean = (1 - a) * self.ema_mean + a * avg
            self.ema_var = (1 - a) * self.ema_var + a * var
    
    def get_adaptive_threshold(self):
        if self.ema_mean is None or self.ema_var is None:
            return 0.5 
        
        ema_std = math.sqrt(max(self.ema_var, 1e-12))
        adaptive_thresh = self.ema_mean + self.std_k * ema_std
        return adaptive_thresh
    
    def input_calibration(self, x, uncertainty_scores):
        """
        x: [batch_size, seq_len, n_var]
        uncertainty_scores: [batch_size]
        """
        self._update_ema_stats(uncertainty_scores)
        
        adaptive_thresh = self.get_adaptive_threshold()
        
        reliable_mask = uncertainty_scores < adaptive_thresh
        unreliable_mask = ~reliable_mask
        
        batch_size = x.shape[0]
        calibrated_x = torch.zeros_like(x)
        
        if reliable_mask.any():
            reliable_indices = torch.where(reliable_mask)[0]
            reliable_x = x[reliable_indices]
            calibrated_reliable = self.expert_reliable.input_calibration(reliable_x)
            calibrated_x[reliable_indices] = calibrated_reliable
        
        if unreliable_mask.any():
            unreliable_indices = torch.where(unreliable_mask)[0]
            unreliable_x = x[unreliable_indices]
            calibrated_unreliable = self.expert_unreliable.input_calibration(unreliable_x)
            calibrated_x[unreliable_indices] = calibrated_unreliable
        
        return calibrated_x, adaptive_thresh, reliable_mask
    
    def output_calibration(self, x, uncertainty_scores):
        """
        x: [batch_size, pred_len, n_var]
        uncertainty_scores: [batch_size]
        """
        adaptive_thresh = self.get_adaptive_threshold()
        
        reliable_mask = uncertainty_scores < adaptive_thresh
        unreliable_mask = ~reliable_mask
        
        batch_size = x.shape[0]
        calibrated_x = torch.zeros_like(x)
        
        if reliable_mask.any():
            reliable_indices = torch.where(reliable_mask)[0]
            reliable_x = x[reliable_indices]
            calibrated_reliable = self.expert_reliable.output_calibration(reliable_x)
            calibrated_x[reliable_indices] = calibrated_reliable
        
        if unreliable_mask.any():
            unreliable_indices = torch.where(unreliable_mask)[0]
            unreliable_x = x[unreliable_indices]
            calibrated_unreliable = self.expert_unreliable.output_calibration(unreliable_x)
            calibrated_x[unreliable_indices] = calibrated_unreliable
        
        return calibrated_x, adaptive_thresh
    
    def get_expert_stats(self):
        return {
            'reliable_expert_params': sum(p.numel() for p in self.expert_reliable.parameters()),
            'unreliable_expert_params': sum(p.numel() for p in self.expert_unreliable.parameters()),
            'total_params': sum(p.numel() for p in self.parameters()),
            'current_threshold': self.get_adaptive_threshold() if self.ema_mean is not None else None
        }


class AdaptiveTester:

    def __init__(self, model, uncertainty_estimator, configs, device):
        self.model = model
        self.uncertainty_estimator = uncertainty_estimator
        self.configs = configs
        self.use_multi_gpu = configs.use_multi_gpu
        self.accelerator = accelerator if self.use_multi_gpu else None

        if  self.use_multi_gpu:
            self.device = self.accelerator.device
        else:
            self.device = device or next(model.parameters()).device
        
        seq_len = self.configs.seq_len_max_irr or self.configs.seq_len
        pred_len = self.configs.pred_len_max_irr or self.configs.pred_len
        enc_in = self.configs.enc_in
        
        self.adaptive_GDC_calibration = AdaptiveGDC_Calibration(
            seq_len=seq_len,
            pred_len=pred_len,
            n_var=enc_in,
            configs = self.configs,
            hidden_dim=configs.cali_hidden_dim,
            var_wise=configs.cali_var_wise
        ).to(self.device)

        self.reliable_expert_optimizer = torch.optim.Adam(
            self.adaptive_GDC_calibration.expert_reliable.parameters(),
            lr=configs.adapt_lr
        )
        
        self.unreliable_expert_optimizer = torch.optim.Adam(
            self.adaptive_GDC_calibration.expert_unreliable.parameters(),
            lr=configs.adapt_lr * 0.5 
        )
        
        self.ue_optimizer = torch.optim.Adam(
            self.uncertainty_estimator.parameters(),
            lr=getattr(configs, 'ue_adapt_lr', configs.adapt_lr * 0.1)
        )
        

        if self.use_multi_gpu:
            self.adaptive_GDC_calibration, self.reliable_expert_optimizer, self.unreliable_expert_optimizer = self.accelerator.prepare(
                self.adaptive_GDC_calibration, self.reliable_expert_optimizer, self.unreliable_expert_optimizer
            )
            self.uncertainty_estimator, self.ue_optimizer = self.accelerator.prepare(
                self.uncertainty_estimator, self.ue_optimizer
            )
            

        
        self.adaptation_steps = 0
        self.trigger_mode = 'ema_percentile'   
        
        self._trg_ema_alpha = getattr(self.configs, '_ema_alpha_trigger', 0.25)  # 0.25 0.5 
        self._trg_std_k = getattr(self.configs, '_std_k_trigger', 0.75) #  0.75 0.5

        self._trg_ema_mean = None
        self._trg_ema_var = None

        
        self._freeze_source_model()
        
        self.GDC_stats = {
            'reliable_samples': 0,
            'unreliable_samples': 0,
            'reliable_updates': 0,
            'unreliable_updates': 0,
            'adaptive_threshold_history': [],  
            'reliable_ratio_history': []  
        }
        
    def _freeze_source_model(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def _unwrap(self, module: nn.Module) -> nn.Module:
        from torch.nn.parallel import DistributedDataParallel as DDP
        return module.module if isinstance(module, DDP) else module

    def adaptation_step(self, batch, uncertainty_scores):
        batch = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}
        
        if not self.should_trigger_adaptation(batch, uncertainty_scores):
            logger.info("不触发更新")
            return 0, 0.0, self.GDC_stats
        
        GDC_cali = self._unwrap(self.adaptive_GDC_calibration) 
        current_threshold = GDC_cali.get_adaptive_threshold()
        
        reliable_mask = uncertainty_scores < current_threshold
        unreliable_mask = ~reliable_mask
        
        reliable_indices = torch.where(reliable_mask)[0]
        unreliable_indices = torch.where(unreliable_mask)[0]
        
        self.GDC_stats['reliable_samples'] += len(reliable_indices)
        self.GDC_stats['unreliable_samples'] += len(unreliable_indices)
        self.GDC_stats['adaptive_threshold_history'].append(current_threshold)
        self.GDC_stats['reliable_ratio_history'].append(len(reliable_indices) / len(uncertainty_scores) if len(uncertainty_scores) > 0 else 0)
        
        
        total_cali_loss = 0.0
        total_ue_loss = 0.0
        num_updates = 0
        
        if len(reliable_indices) > 0:
            self.model.train()
            reliable_loss = self._update_expert(
                expert_idx=0, 
                batch=batch,
                indices=reliable_indices,
                expert_optimizer=self.reliable_expert_optimizer
            )
            self.model.eval()
            total_cali_loss += reliable_loss
            self.GDC_stats['reliable_updates'] += 1
            num_updates += 1
        
        if len(unreliable_indices) > 0:
            self.model.train()
            unreliable_loss = self._update_expert(
                expert_idx=1,  
                batch=batch,
                indices=unreliable_indices,
                expert_optimizer=self.unreliable_expert_optimizer
            )
            self.model.eval()
            total_cali_loss += unreliable_loss
            self.GDC_stats['unreliable_updates'] += 1
            num_updates += 1
        
        if len(reliable_indices) > 0:
            self.model.train()
            ue_loss = self._update_uncertainty_estimator(batch, reliable_indices)
            self.model.eval()
            total_ue_loss += ue_loss
        
        self.adaptation_steps += 1
        
        if num_updates > 0:
            avg_cali_loss = total_cali_loss / num_updates
            return (len(reliable_indices) + len(unreliable_indices)), avg_cali_loss, self.GDC_stats

        else:
            return 0, 0.0, self.GDC_stats
    
    def _update_expert(self, expert_idx, batch, indices, expert_optimizer):
        expert_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                expert_batch[key] = value[indices].to(self.device)
            else:
                expert_batch[key] = value
        
        x_batch = expert_batch["x"]
        
        GDC_cali = self._unwrap(self.adaptive_GDC_calibration)  
        if expert_idx == 0:
            expert = GDC_cali.expert_reliable
        else:
            expert = GDC_cali.expert_unreliable
        
        expert.train()
        
        expert_loss = 0.0
        
        for _ in range(5):  
            expert_optimizer.zero_grad()
            
            calibrated_x = expert.input_calibration(x_batch)
            calibrated_batch = expert_batch.copy()
            calibrated_batch["x"] = calibrated_x
            
            model_outputs = self.model(**calibrated_batch, exp_stage="test")
            
            calibrated_pred = expert.output_calibration(model_outputs["pred"])
            
            true_batch = model_outputs["true"]
            mask_batch = model_outputs.get("mask", None)
            

            diff = calibrated_pred - true_batch
            if mask_batch is not None:
                num = ((diff ** 2) * mask_batch).sum(dim=[1, 2])
                den = mask_batch.sum(dim=[1, 2]).clamp_min(1e-8)
                per_sample_cali = num / den
            else:
                per_sample_cali = (diff ** 2).mean(dim=[1, 2])
            loss = per_sample_cali.mean()
            
            if self.use_multi_gpu:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            
            clip_grad_norm_(expert.parameters(), 1.0)
            expert_optimizer.step()
            
            expert_loss += loss.item()
        
        expert.eval()
        
        return expert_loss / 5 
    
    def _update_uncertainty_estimator(self, batch, indices):
        ue_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                ue_batch[key] = value[indices].to(self.device)
            else:
                ue_batch[key] = value
        
        x_batch = ue_batch["x"]
        
        self.uncertainty_estimator.train()
        
        ue_loss = 0.0
        
        for _ in range(5):
            self.ue_optimizer.zero_grad()
            
            with torch.no_grad():
                GDC_cali = self._unwrap(self.adaptive_GDC_calibration) 
                calibrated_x = GDC_cali.expert_reliable.input_calibration(x_batch)
                calibrated_batch = ue_batch.copy()
                calibrated_batch["x"] = calibrated_x
                model_outputs = self.model(**calibrated_batch, exp_stage="test")
                calibrated_pred = GDC_cali.expert_reliable.output_calibration(model_outputs["pred"])
            
            true_batch = model_outputs["true"]
            mask_batch = model_outputs["mask"]
            
            if mask_batch is not None:
                true_errors = F.mse_loss(
                    calibrated_pred * mask_batch,  

                    true_batch * mask_batch,
                    reduction='none'
                ).mean(dim=[1, 2])
            else:
                true_errors = F.mse_loss(calibrated_pred, true_batch, reduction='none')

                true_errors = true_errors.mean(dim=[1, 2])
            
            ue_mod = self._unwrap(self.uncertainty_estimator)
            denom = (ue_mod.error_max - ue_mod.error_min).clamp_min(1e-8)
            true_errors_norm = (true_errors - ue_mod.error_min) / denom
            true_errors_norm = true_errors_norm.clamp(0.0, 1.0)
            
            ue_predictions = ue_mod(x_batch.detach(), calibrated_pred.detach())
            per_sample_ue = F.l1_loss(ue_predictions, true_errors_norm, reduction='none')
            loss = per_sample_ue.mean()
            

            if self.use_multi_gpu:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            
            clip_grad_norm_(self.uncertainty_estimator.parameters(), 1.0)
            self.ue_optimizer.step()
            
            ue_loss += loss.item()
        
        self.uncertainty_estimator.eval()
        
        return ue_loss / 5

    def should_trigger_adaptation(self, batch, uncertainty_scores):

        if isinstance(uncertainty_scores, torch.Tensor):
            u = uncertainty_scores.detach().cpu().flatten()
        else:
            u = torch.as_tensor(uncertainty_scores)

        if self.trigger_mode == 'ema_percentile':
            avg = u.mean().item()   
            var = u.var(unbiased=False).item() 
            if self._trg_ema_mean is None:
                self._trg_ema_mean = avg
                self._trg_ema_var = var
            else:
                a = self._trg_ema_alpha
                self._trg_ema_mean = (1 - a) * self._trg_ema_mean + a * avg
                self._trg_ema_var = (1 - a) * self._trg_ema_var + a * var
            
            ema_std = math.sqrt(max(self._trg_ema_var, 1e-12))
            ema_thresh = self._trg_ema_mean + self._trg_std_k * ema_std
            
            dyn_thresh = ema_thresh
            return (avg > dyn_thresh) or (u.max().item() > dyn_thresh)

        return True
    
    def forward(self, batch, test=False):
        if next(self.model.parameters()).device != batch["x"].device:
            batch = {k: (v.to(next(self.model.parameters()).device) if isinstance(v, torch.Tensor) else v) 
                    for k, v in batch.items()}

        with torch.no_grad():

            GDC_cali = self._unwrap(self.adaptive_GDC_calibration)  
            calibrated_x = GDC_cali.expert_reliable.input_calibration(batch["x"])
            calibrated_batch = batch.copy()
            calibrated_batch["x"] = calibrated_x
            model_outputs = self.model(**calibrated_batch, exp_stage="test")
            calibrated_pred = GDC_cali.expert_reliable.output_calibration(model_outputs["pred"])

        
            uncertainty_scores = self.uncertainty_estimator(batch["x"], calibrated_pred)

        
        GDC_cali = self._unwrap(self.adaptive_GDC_calibration)  
        calibrated_x, adaptive_thresh, reliable_mask = GDC_cali.input_calibration(
            batch["x"], uncertainty_scores
        )
        calibrated_batch = batch.copy()
        calibrated_batch["x"] = calibrated_x
        

        outputs = self.model(**calibrated_batch, exp_stage="test")
        

        calibrated_pred, _ = GDC_cali.output_calibration(
            outputs["pred"], uncertainty_scores
        )
        outputs["pred"] = calibrated_pred
        
        if test:
            
            # wo_cali = F.mse_loss(
            #     outputs_model["pred"] * outputs_model["mask"],
            #     outputs_model["true"] * outputs_model["mask"],
            #     reduction='none'
            # ).mean(dim=[1, 2])
            wo_cali = F.mse_loss(
                model_outputs["pred"] * model_outputs["mask"],
                model_outputs["true"] * model_outputs["mask"],
                reduction='none'
            ).mean(dim=[1, 2])
            
            w_cali = F.mse_loss(
                outputs["pred"] * outputs["mask"],
                outputs["true"] * outputs["mask"],
                reduction='none'
            ).mean(dim=[1, 2])
            
            return outputs, uncertainty_scores, wo_cali, w_cali, adaptive_thresh, reliable_mask
        else:
            return outputs, uncertainty_scores
    
    def process_batch(self, batch):
        outputs = None
        uncertainty = None
        adaptation_triggered = False
        n_updated = 0
        adaptation_loss = None
        wo_cali = None
        GDC_stats = {
                'reliable_samples': 0,
                'unreliable_samples': 0,
                'total_samples': 0,
            }
        adaptive_thresh = None

        batch = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}
        
        outputs, uncertainty, wo_cali, w_cali, adaptive_thresh, reliable_mask = self.forward(batch, test=True)
        
        adaptation_triggered = False
        n_updated = 0
        adaptation_loss = 0.0
        
        if self.should_trigger_adaptation(batch, uncertainty):
            n_updated, adaptation_loss, GDC_stats = self.adaptation_step(batch, uncertainty)
            adaptation_triggered = (n_updated > 0)

        else:
            adaptive_thresh = None
            GDC_stats = {
                'reliable_samples': 0,
                'total_samples': 0,
            }
        
        
        return outputs, uncertainty, adaptation_triggered, GDC_stats


class GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True):
        super(GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        
        if var_wise:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len))

        self.mlp = nn.Sequential(
            nn.Linear(window_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, window_len)
        )
        self.var_mlp = nn.Sequential(
            nn.Linear(n_var, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_var)
        )

        self.weight.data.zero_()
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))

    def forward(self, x):
        # x shape: [batch_size, seq_len, n_features]
        if self.var_wise:
            calibrated = x + torch.tanh(self.gating) * (torch.einsum('biv,iov->bov', x, self.weight) + self.bias) 
        else:
            calibrated = x + torch.tanh(self.gating) * (torch.einsum('biv,io->bov', x, self.weight) + self.bias)
        
        calibrated = x + torch.tanh(self.gating) * self.mlp(calibrated.transpose(1, 2)).transpose(1, 2)
        
        return calibrated

class Calibration(nn.Module):
    def __init__(self, seq_len, pred_len, n_var, hidden_dim=64, gating_init=0.01, var_wise=True):
        super(Calibration, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_var = n_var
        
        self.in_cali = GCM(seq_len, n_var, hidden_dim, gating_init, var_wise)
        self.out_cali = GCM(pred_len, n_var, hidden_dim, gating_init, var_wise)
        
    def input_calibration(self, x):
        return self.in_cali(x)
    
    def output_calibration(self, x):
        return self.out_cali(x)


class Exp_Main(Exp_Basic):
    def __init__(self, configs: ExpConfigs):
        super(Exp_Main, self).__init__(configs)
        self.uncertainty_estimators = {}  

    def _build_model(self) -> Module:
        # dynamically import the desired model class
        model_module = importlib.import_module("models." + self.configs.model_name)
        model = model_module.Model(self.configs)
        return model

    def _get_data(self, flag: str) -> tuple[Dataset, DataLoader]:
        data_set, data_loader = data_provider(self.configs, flag)
        return data_set, data_loader

    def _select_optimizer(self, model: Module) -> optim.Optimizer:
        model_optim = optim.Adam(model.parameters(), lr=self.configs.learning_rate)
        return model_optim

    def _select_lr_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler:
        # Initialize scheduler based on configs.lradj
        if self.configs.lr_scheduler == 'ExponentialDecayLR':
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
        elif self.configs.lr_scheduler == 'ManualMilestonesLR':
            from lr_schedulers.ManualMilestonesLR import ManualMilestonesLR
            user_milestones = {2:5e-5, 4:1e-5, 6:5e-6, 8:1e-6, 10:5e-7, 15:1e-7, 20:5e-8}
            milestones = {k-1: v for k, v in user_milestones.items()}
            scheduler = ManualMilestonesLR(optimizer, milestones)
        elif self.configs.lr_scheduler == 'DelayedStepDecayLR':
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch < 2 else (0.8 ** (epoch - 2)))
        elif self.configs.lr_scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.configs.train_epochs, eta_min=0.0)
        elif self.configs.lr_scheduler == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[0.75 * self.configs.train_epochs, 0.9 * self.configs.train_epochs], gamma=self.configs.lr_scheduler_gamma
            )
        else:
            logger.exception(f"Unknown lr scheduler '{self.configs.lr_scheduler}'", stack_info=True)
            exit(1)

        return scheduler

    def _select_criterion(self) -> Module:
        # dynamically import the desired loss function
        loss_module = importlib.import_module("loss_fns." + self.configs.loss)
        criterion = loss_module.Loss(self.configs)
        return criterion

    def _get_state_dict(self, path: Path) -> OrderedDict:
        '''
        Fix model state dict errors
        '''
        logger.info(f"Loading model checkpoint from {path}")
        state_dict = torch.load(path, map_location=f"cuda:{self.configs.gpu_id}" if self.configs.use_gpu else "cpu")
        new_state_dict = OrderedDict()
        if_fixed = False
        for key, value in state_dict.items():
            # you may insert modifications to the key and value here
            if 's4' in key and (('B' in key or 'P' in key or 'w' in key) and ('weight' not in key)):
                # S4 layer don't need to load these weights
                if_fixed = True
                continue
            new_state_dict[key] = value.contiguous()
        if if_fixed:
            logger.warning("Automatically fixed model state dict errors. It may cause unexpected behavior!")
        return new_state_dict

    def _check_model_outputs(self, batch:dict, outputs:dict) -> None:
        '''
        Perform necessary checks on model's outputs
        '''
        # check if the data type is dict
        if type(outputs) is not dict:
            logger.exception(f"Expect model's forward function to return dict. Current output's data type is {type(outputs)}.", stack_info=True)
            exit(1)

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            # check if outputs' true is the the same as input dataset's y
            if "true" in outputs.keys() and not torch.equal(batch["y"], outputs["true"]):
                logger.warning(f"Model's outputs['true'] is not equal to input's batch['y']. Please confirm that you are not using input's batch['y'] as ground truth. This is expected in some models such as diffusion.")

    def _merge_gathered_dicts(self, dicts: list[dict]) -> dict:
        '''
        manually merge list of dictionary gathered when testing
        accelerate.gather_for_metrics may have unexpected behavior, thus merge manually instead
        '''
        merged_dict = {}
        keys_not_returned = []
        for d in dicts:
            for key, tensor in d.items():
                if type(tensor).__name__ != "Tensor":
                    # skip value that is not Pytorch Tensor
                    if key not in keys_not_returned:
                        keys_not_returned.append(key)
                        logger.warning(f"{key=} will not be gathered for metric calculation in test, since its value has data type '{type(tensor).__name__}', which is not 'Tensor'")
                    continue
                if key in merged_dict:
                    merged_dict[key] = torch.cat((merged_dict[key], tensor.detach().cpu()), dim=0)
                else:
                    merged_dict[key] = tensor.detach().cpu()
        return merged_dict
    
    def _prepare_for_gpu_gather(self, obj, device):
        """
        Recursively move tensors to GPU and make them contiguous for accelerator.gather/gather_for_metrics.
        Supports dict, list, tuple, and tensors. Non-tensor types are returned as-is.
        """
        if isinstance(obj, torch.Tensor):
            if obj.is_sparse:
                obj = obj.to_dense()
            if obj.device != device:
                obj = obj.to(device)
            if not obj.is_contiguous():
                obj = obj.contiguous()
            return obj
        elif isinstance(obj, dict):
            return {k: self._prepare_for_gpu_gather(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            converted = [self._prepare_for_gpu_gather(v, device) for v in obj]
            return type(obj)(converted)
        else:
            return obj

    def vali(self, model_train: Module, vali_loader: DataLoader, criterion: Module, current_epoch: int, train_stage: int) -> np.ndarray:
        total_loss = []
        model_train.eval()
        with torch.no_grad():
            with tqdm(total=len(vali_loader), leave=False, desc="Validating") as it:
                batch: dict[str, Tensor] # type hints
                for i, batch in enumerate(vali_loader):
                    # warn if the size does not match
                    if batch[next(iter(batch))].shape[0] != self.configs.batch_size and current_epoch == 0:
                        logger.warning(f"Batch No.{i} of total {len(vali_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}")
                    if "y_mask" in batch.keys():
                        if torch.sum(batch["y_mask"]).item() == 0:
                            if current_epoch == 0:
                                logger.warning(f"Batch No.{i} of total {len(vali_loader)} has no evaluation point (inferred from y_mask), thus skipping")
                            continue

                    device = accelerator.device if self.configs.use_multi_gpu else torch.device(f"cuda:{self.configs.gpu_id}")
                    batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

                    # some model's forward function return different values in "train", "val", "test", they can use `exp_stage` as argument to distinguish
                    outputs: dict[str, Tensor] = model_train(
                        exp_stage="val",
                        train_stage=train_stage,
                        **batch,
                    )

                    loss: Tensor = criterion(
                        exp_stage="val",
                        model=model_train,
                        **outputs
                    )["loss"]
                    total_loss.append(loss.item())

                    if accelerator.is_main_process:
                        # update only in main process
                        it.update()
                        it.set_postfix(loss=f"{loss.item():.2e}")
        total_loss = np.average(total_loss)
        model_train.train()
        return total_loss

    def train(self) -> None:
        logger.info('>>>>>>> training start <<<<<<<')
        path = Path(self.configs.checkpoints) / self.configs.dataset_name / self.configs.model_name / self.configs.model_id / f"{self.configs.seq_len}_{self.configs.pred_len}" / self.configs.subfolder_train / f"iter{self.configs.itr_i}"
        if (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
            import wandb

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # model initialized after dataset to obtain possible dynamic information from dataset (e.g., seq_len_max_irr)
        model_train = self._build_model()

        model_optim = self._select_optimizer(model_train)
        lr_scheduler = self._select_lr_scheduler(model_optim)
        criterion = self._select_criterion()

        if self.configs.use_multi_gpu:
            train_loader, vali_loader, model_train, model_optim = accelerator.prepare(
                train_loader, vali_loader, model_train, model_optim
            )
            accelerator.register_for_checkpointing(model_optim)
            device_runtime = accelerator.device

        if not self.configs.use_multi_gpu:
            device_runtime = torch.device(f"cuda:{self.configs.gpu_id}" if self.configs.use_gpu else "cpu")
            model_train = model_train.to(device_runtime)

        # Save initial states
        initial_optimizer_state = model_optim.state_dict()
        initial_scheduler_state = lr_scheduler.state_dict()

        if_nan_loss = False # break nested loop without using for...else...
        for train_stage in range(1, self.configs.n_train_stages + 1):
            early_stopping = EarlyStopping(patience=self.configs.patience, verbose=True)
            logger.info(f"Train stage {train_stage}/{self.configs.n_train_stages} starts.")
            for epoch in tqdm(range(self.configs.train_epochs), desc="Epochs"):
                train_loss = []
                model_train.train()
                with tqdm(total=len(train_loader), leave=False, desc="Training") as it:
                    batch: dict[str, Tensor] # type hints
                    for i, batch in enumerate(train_loader):
                        # warn if the size does not match
                        if batch[next(iter(batch))].shape[0] != self.configs.batch_size and epoch == 0:
                            logger.warning(f"Batch No.{i} of total {len(train_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}")
                        if "y_mask" in batch.keys():
                            if torch.sum(batch["y_mask"]).item() == 0:
                                if epoch == 0:
                                    logger.warning(f"Batch No.{i} of total {len(train_loader)} has no evaluation point (inferred from y_mask), thus skipping")
                                continue
                        model_optim.zero_grad()
                        if not self.configs.use_multi_gpu:
                            batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}

                        outputs: dict[str, Tensor] = model_train(
                            exp_stage="train",
                            train_stage=train_stage,
                            current_epoch=epoch,
                            **batch,
                        )

                        # check model's outputs only in the first iteration
                        if i == 0 and epoch == 0:
                            self._check_model_outputs(batch, outputs)
                        
                        loss: Tensor = criterion(
                            exp_stage="train",
                            model=model_train,
                            **outputs
                        )["loss"]

                        # check loss
                        if torch.any(torch.isnan(loss)):
                            logger.exception("Loss is nan! Training interruptted!")
                            for key, value in outputs.items():
                                if key == "loss":
                                    continue
                                elif type(value).__name__ != "Tensor" and torch.any(torch.isnan(value)):
                                    logger.error(f"Nan value found in model's output tensor '{key}' of shape {value.shape}: {value}")
                            logger.info("Hint: possible cause for nan loss: 1. large learning rate; 2. sqrt(0); 3. ReLU->LeakyReLU")
                            if_nan_loss = True
                            break

                        train_loss.append(loss.item())

                        if accelerator.is_main_process:
                            # update progress bar only in main process
                            it.update()
                            it.set_postfix(loss=f"{loss.item():.2e}")

                        if self.configs.sweep:
                            loss.backward(retain_graph=self.configs.retain_graph)
                        else:
                            accelerator.backward(loss, retain_graph=self.configs.retain_graph)
                        model_optim.step()

                if if_nan_loss:
                    accelerator.set_trigger()
                    if accelerator.check_trigger():
                        accelerator.wait_for_everyone()
                        break

                # validation
                if epoch % self.configs.val_interval == 0:
                    vali_loss = self.vali(
                        model_train=model_train, 
                        vali_loader=vali_loader, 
                        criterion=criterion, 
                        current_epoch=epoch,
                        train_stage=train_stage
                    )
                    early_stopping(vali_loss, model_train, path)
                    if (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
                        wandb.log({
                            "loss_train": np.mean(train_loss),
                            "loss_val": vali_loss,
                            "loss_val_best": -early_stopping.best_score
                        })
                    if early_stopping.early_stop:
                        logger.info("Early stopping")
                        accelerator.set_trigger()
                elif (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
                    wandb.log({
                        "loss_train": np.mean(train_loss),
                    })

                lr_scheduler.step()
                logger.debug(f'Updating learning rate to {lr_scheduler.get_last_lr()[0]:.6e}')
                if accelerator.check_trigger():
                    accelerator.wait_for_everyone()
                    break

            # Reset optimizer, scheduler
            model_optim.load_state_dict(initial_optimizer_state)
            lr_scheduler.load_state_dict(initial_scheduler_state)

    def _setup_uncertainty_paths(self, checkpoint_location_itr, iter_id=None):
        self.configs.trained_model_path = checkpoint_location_itr / "pytorch_model.bin"
        
        if iter_id is not None:
            uncertainty_filename = f"uncertainty_estimator_iter{iter_id}.pth"
        else:
            uncertainty_filename = "uncertainty_estimator.pth"
        
        self.configs.uncertainty_model_save_path = checkpoint_location_itr / uncertainty_filename
        self.configs.uncertainty_model_path = checkpoint_location_itr / uncertainty_filename
        
    
    def _find_latest_checkpoint(self):
        checkpoint_location = Path(self.configs.checkpoints) / self.configs.dataset_name / self.configs.model_name / self.configs.model_id / f"{self.configs.seq_len}_{self.configs.pred_len}"
        
        if self.configs.load_checkpoints_test:
            try:
                
                child_folders = [(entry.name, entry) for entry in checkpoint_location.iterdir() if entry.is_dir()]
                if len(child_folders) == 0:
                    logger.exception(f"No folder under '{checkpoint_location}' matches the model_id '{self.configs.model_id}'.", stack_info=True)
                    return None, None
                latest_folder = sorted(child_folders, key=lambda item: datetime.datetime.strptime(item[0], "%Y_%m%d_%H%M"))[-1][1].name
                checkpoint_location = checkpoint_location / latest_folder
                self.configs.subfolder_train = latest_folder
                
                actual_itrs = len([entry.name for entry in checkpoint_location.iterdir() if entry.is_dir()])
                checkpoint_location_itr = checkpoint_location / f"iter{actual_itrs-1}"  
                return checkpoint_location, checkpoint_location_itr
                
            except Exception as e:
                logger.exception(f"{e}", stack_info=True)
                return None, None
        else:
            train_folder = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
            path = checkpoint_location / train_folder / f"iter0"
            path.mkdir(parents=True, exist_ok=True)
            checkpoint_location = checkpoint_location / train_folder
            self.configs.subfolder_train = train_folder
            return checkpoint_location, path
        
    def _unwrap(self, module: nn.Module) -> nn.Module:
        from torch.nn.parallel import DistributedDataParallel as DDP
        return module.module if isinstance(module, DDP) else module
    
    def train_uncertainty_estimator_for_all_iters(self, checkpoint_location=None):
        
        if checkpoint_location is None:
            checkpoint_location, _ = self._find_latest_checkpoint()
            if checkpoint_location is None:
                return None
        iter_folders = self._get_all_iteration_folders(checkpoint_location)
        if not iter_folders:
            return None
        
        
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')

        all_uncertainty_estimators = {}
        
        for iter_folder in iter_folders:
            iter_id = self._extract_iter_id(iter_folder.name)
            
            self._setup_uncertainty_paths(iter_folder, iter_id)
            
            if self._check_uncertainty_model_exists(iter_folder, iter_id):
                uncertainty_estimator = self._load_uncertainty_estimator(iter_id)
                if uncertainty_estimator is not None:
                    all_uncertainty_estimators[iter_id] = uncertainty_estimator
                continue
            
            model = self._build_model().eval()
            if self.configs.trained_model_path.exists():
                checkpoint = torch.load(self.configs.trained_model_path, map_location=self.device)
                
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        name = k[7:]
                    else:
                        name = k
                    new_state_dict[name] = v
                
                model.load_state_dict(new_state_dict, strict=False)
            else:
                continue
            
            uncertainty_estimator = self._train_single_uncertainty_estimator(
                self.configs, train_loader, val_loader, model, self.device, iter_id
            )
            
            if uncertainty_estimator is not None:
                torch.save(
                    uncertainty_estimator.state_dict(),
                    self.configs.uncertainty_model_save_path
                )
                all_uncertainty_estimators[iter_id] = uncertainty_estimator
        
        return all_uncertainty_estimators

    def train_uncertainty_estimator_for_current_iter(self, checkpoint_location_itr=None, iter_id=None):
        if iter_id is None:
            iter_id = self.configs.itr_i
        
        if checkpoint_location_itr is None:
            checkpoint_location, _ = self._find_latest_checkpoint()
            if checkpoint_location is None:
                return None
            checkpoint_location_itr = checkpoint_location / f"iter{iter_id}"
        
        self._setup_uncertainty_paths(checkpoint_location_itr, iter_id)
        
        if self._check_uncertainty_model_exists(checkpoint_location_itr, iter_id):
            return self._load_uncertainty_estimator(iter_id)
        
        train_data, train_loader = self._get_data(flag='train')
        
        model = self._build_model().eval()
        if self.configs.trained_model_path.exists():
            checkpoint = torch.load(self.configs.trained_model_path, map_location=self.device)
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            return None

        uncertainty_estimator = self._train_single_uncertainty_estimator(
            self.configs, train_loader, model, self.device, iter_id
        )
        
        if uncertainty_estimator is not None:
            torch.save(
                uncertainty_estimator.state_dict(),
                self.configs.uncertainty_model_save_path
            )
            self.uncertainty_estimators[iter_id] = uncertainty_estimator
        
        return uncertainty_estimator
    
    def _get_all_iteration_folders(self, checkpoint_location):
        iteration_folders = []
        for item in checkpoint_location.iterdir():
            if item.is_dir() and item.name.startswith("iter"):
                iteration_folders.append(item)
        
        iteration_folders.sort(key=lambda x: self._extract_iter_id(x.name))
        return iteration_folders

    def _extract_iter_id(self, folder_name):
        try:
            return int(folder_name.replace("iter", ""))
        except ValueError:
            return 0

    def _check_uncertainty_model_exists(self, checkpoint_location_itr, iter_id=None):
        if iter_id is not None:
            uncertainty_model_path = checkpoint_location_itr / f"uncertainty_estimator_iter{iter_id}.pth"
        else:
            uncertainty_model_path = checkpoint_location_itr / "uncertainty_estimator.pth"
        return uncertainty_model_path.exists()

    def _train_single_uncertainty_estimator(self, configs, train_loader, val_loader, model, device, iter_id=None):
        def _collect_data_for_estimator(data_loader, model, uncertainty_estimator, is_train=True):
            all_predictions = []
            all_targets = []
            all_uncertainty_inputs = []
            all_y_masks = []
            
            model.eval()  
            with torch.no_grad():
                for batch in tqdm(data_loader, desc=f"{iter_id}"):
                    if not configs.use_multi_gpu:
                        batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # model = model.cpu()
                    model = model.to(device)
                    outputs = model(**batch, exp_stage = "test")
                    pred = outputs["pred"]
                    true = outputs["true"]
                    y_mask = outputs["mask"]
                    
                    all_predictions.append(pred.detach().cpu())
                    all_targets.append(true.detach().cpu())
                    all_uncertainty_inputs.append(batch["x"].detach().cpu())
                    all_y_masks.append(y_mask.detach().cpu())
                    
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_uncertainty_inputs = torch.cat(all_uncertainty_inputs, dim=0)
            all_y_masks = torch.cat(all_y_masks, dim=0)
            
            # ||ŷ - y||₂²
            prediction_errors = F.mse_loss(all_predictions * all_y_masks, all_targets * all_y_masks, reduction='none')
            prediction_errors = prediction_errors.mean(dim=[1, 2])  # [num_samples]
            
            # 0-1
            if is_train:
                error_min = prediction_errors.min()
                error_max = prediction_errors.max()
        
                ue_mod = self._unwrap(uncertainty_estimator)
                ue_mod.error_min.data = error_min.to(ue_mod.error_min.device)
                ue_mod.error_max.data = error_max.to(ue_mod.error_max.device)
                normalized_errors = (prediction_errors - error_min) / (error_max - error_min + 1e-8)
            
            else:
                
                ue_mod = self._unwrap(uncertainty_estimator)
                normalized_errors = \
                    (prediction_errors - ue_mod.error_min.cpu()) / (ue_mod.error_max.cpu() - ue_mod.error_min.cpu() + 1e-8)


            return all_uncertainty_inputs, all_predictions, normalized_errors
            

        uncertainty_estimator = OfflineUncertaintyEstimator(
            seq_len=configs.seq_len_max_irr or configs.seq_len,
            pred_len=configs.pred_len_max_irr or configs.pred_len,
            enc_in=configs.enc_in,
            hidden_dim=getattr(configs, 'uncertainty_hidden_dim', 128)
        ).to(device)

        train_inputs, train_preds, train_errors = _collect_data_for_estimator(train_loader, model, uncertainty_estimator, is_train=True)
        val_inputs, val_preds, val_errors = _collect_data_for_estimator(val_loader, model, uncertainty_estimator, is_train=False)
        
        optimizer = torch.optim.Adam(
            uncertainty_estimator.parameters(),
            lr=getattr(configs, 'uncertainty_lr', 1e-5)
        )
        
        criterion = nn.L1Loss()  
        
        uncertainty_estimator.train()
               
        train_dataset = torch.utils.data.TensorDataset(
            train_inputs, train_preds, train_errors
        )
        val_dataset = torch.utils.data.TensorDataset(
            val_inputs, val_preds, val_errors
        )

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=configs.batch_size,
            shuffle=True
        )
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=configs.batch_size,
            shuffle=True
        )
        
        num_epochs = getattr(configs, 'uncertainty_epochs', 300)
        train_loss_list, val_loss_list = [], []
        # Early stopping configuration
        ue_patience = getattr(configs, 'uncertainty_patience', 10)  
        best_val_loss = float('inf')
        best_state_dict = None
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for x_batch, pred_batch, error_batch in train_data_loader:
                x_batch = x_batch.to(device)
                pred_batch = pred_batch.to(device)
                error_batch = error_batch.to(device)
                
                predicted_uncertainty = uncertainty_estimator(x_batch, pred_batch)
                
                loss = criterion(predicted_uncertainty, error_batch)
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_train_loss = total_loss / num_batches
                train_loss_list.append(avg_train_loss)

                # eval
                total_val_loss = 0
                for x_batch, pred_batch, error_batch in val_data_loader:
                    x_batch = x_batch.to(device)
                    pred_batch = pred_batch.to(device)
                    error_batch = error_batch.to(device)
                    
                    with torch.no_grad():
                        predicted_uncertainty = uncertainty_estimator(x_batch, pred_batch)
                        val_loss = criterion(predicted_uncertainty, error_batch)
                    
                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_data_loader)
                val_loss_list.append(avg_val_loss)
    
                # Early stopping check
                if avg_val_loss < best_val_loss - 1e-8:  
                    best_val_loss = avg_val_loss
                    best_state_dict = copy.deepcopy(uncertainty_estimator.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 10                   
                    if epochs_no_improve >= ue_patience:
                        break

        if best_state_dict is not None:
            uncertainty_estimator.load_state_dict(best_state_dict)

        return uncertainty_estimator
    
    def _load_uncertainty_estimator(self, iter_id=None):
        if iter_id is not None:
            uncertainty_model_path = self.configs.uncertainty_model_path.parent / f"uncertainty_estimator_iter{iter_id}.pth"
        else:
            uncertainty_model_path = self.configs.uncertainty_model_path
        
        if uncertainty_model_path.exists():
            uncertainty_estimator = OfflineUncertaintyEstimator(
                seq_len=self.configs.seq_len_max_irr or self.configs.seq_len,
                pred_len=self.configs.pred_len_max_irr or self.configs.pred_len,
                enc_in=self.configs.enc_in,
                hidden_dim=getattr(self.configs, 'uncertainty_hidden_dim', 128)
            ).to(self.device)
            
            uncertainty_estimator.load_state_dict(
                torch.load(uncertainty_model_path, map_location=self.device)
            )
            uncertainty_estimator.eval()
            return uncertainty_estimator
        else:
            return None

    def load_uncertainty_estimator_for_iter(self, checkpoint_location_itr, iter_id):
        self._setup_uncertainty_paths(checkpoint_location_itr, iter_id)
        return self._load_uncertainty_estimator(iter_id)
    
    def train_uncertainty_only_for_all_iters(self):
        checkpoint_location, _ = self._find_latest_checkpoint()
        if checkpoint_location is None:
            return
        uncertainty_estimators = self.train_uncertainty_estimator_for_all_iters(checkpoint_location)
        

    def _check_any_uncertainty_model_exists(self):
        checkpoint_location, checkpoint_location_itr = self._find_latest_checkpoint()
        if not checkpoint_location:
            return False
        
        all_iter_folders = self._get_all_iteration_folders(checkpoint_location)
        for iter_folder in all_iter_folders:
            iter_id = self._extract_iter_id(iter_folder.name)
            if self._check_uncertainty_model_exists(iter_folder, iter_id):
                return True
        
        return False
    
    def standard_test(self, model_test, test_loader, folder_path):
        
        # dictionary holding input and output data
        array_dict: dict[str, list[np.ndarray] | np.ndarray] = {}
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            input_tensor_names = ["x", "y", "x_mask", "y_mask", "sample_ID"]
            output_tensor_names = ["pred"]
        else:
            raise NotImplementedError

        for tensor_name in input_tensor_names + output_tensor_names:
            array_dict[tensor_name] = []

        # try to recover from cache saved by save_cache_arrays, if any
        cache_folder = folder_path / "cache_standard"
        n_cache_batches = 0
        if cache_folder.exists():
            logger.warning(f"Trying to recover the standard testing process using cache files in {cache_folder}")
            for tensor_name in output_tensor_names:
                cache_file_path = cache_folder / f"output_{tensor_name}.npy"
                if cache_file_path.exists():
                    cache_array = np.load(cache_file_path)
                    n_cache_samples = cache_array.shape[0]
                    # overwrite init content with cache
                    array_dict[tensor_name] = [cache_array[i:i + self.configs.batch_size] for i in range(0, n_cache_samples, self.configs.batch_size)] # ndarray -> list[ndarray]
                else:
                    logger.error(f"Cache file for {tensor_name} not found. You may encounter unexpected error if proceed!")
            n_cache_batches = len(array_dict[tensor_name])

        with torch.no_grad():
            batch: dict[str, Tensor] # type hints
            for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), leave=False, desc="Standard Test"):
                if n_cache_batches > 0:
                    # recovering from cache. append input batch and skip it, such that model don't have to inference again.
                    batch_all: list[dict] = accelerator.gather_for_metrics([batch])
                    batch_all: dict = self._merge_gathered_dicts(batch_all)
                    for tensor_name in input_tensor_names:
                        if tensor_name in batch_all.keys():
                            array_dict[tensor_name].append(batch_all[tensor_name].detach().cpu().numpy())
                    n_cache_batches -= 1
                    continue
                # warn if the size does not match
                if batch[next(iter(batch))].shape[0] != self.configs.batch_size:
                    logger.warning(f"Batch No.{i} of total {len(test_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}")
                    # continue
                if not self.configs.use_multi_gpu:
                    batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}

                outputs: dict[str, Tensor] = model_test(
                    exp_stage="test",
                    **batch,
                )

                # check model's outputs only in the first iteration
                if i == 0:
                    self._check_model_outputs(batch, outputs)

                batch_all: list[dict] = accelerator.gather_for_metrics([batch])
                batch_all: dict = self._merge_gathered_dicts(batch_all)
                outputs_all: list[dict] = accelerator.gather_for_metrics([outputs])
                outputs_all: dict = self._merge_gathered_dicts(outputs_all)

                for tensor_name in input_tensor_names:
                    if tensor_name in batch_all.keys():
                        array_dict[tensor_name].append(batch_all[tensor_name].detach().cpu().numpy())
                for tensor_name in output_tensor_names:
                    if tensor_name in outputs_all.keys():
                        array_dict[tensor_name].append(outputs_all[tensor_name].detach().cpu().numpy())

                if self.configs.save_cache_arrays:
                    # save intermediate model outputs, to enable recovery from interruption
                    cache_folder.mkdir(exist_ok=True)
                    for tensor_name in output_tensor_names:
                        if len(array_dict[tensor_name]) > 0:
                            np.save(
                                cache_folder / f"output_{tensor_name}.npy",
                                np.concatenate(array_dict[tensor_name], axis=0)
                            )
                    logger.debug(f"Model outputs saved into cache folder {cache_folder}")

        for tensor_name in input_tensor_names + output_tensor_names:
            if len(array_dict[tensor_name]) > 0:
                array_dict[tensor_name] = np.concatenate(array_dict[tensor_name], axis=0)
            else:
                array_dict[tensor_name] = None # reset to default value for metric calculation

        metrics_standard = None
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            metrics_standard = metric(**array_dict)
            if (self.configs.wandb and accelerator.is_main_process and self.configs.is_training) or self.configs.sweep:
                import wandb
                wandb.log({
                    "loss_test": np.mean(metrics_standard["MSE"]),
                })
        
        if metrics_standard is not None:
            # convert to float before saving to json
            for key, value in metrics_standard.items():
                if isinstance(value, np.float32):
                    metrics_standard[key] = float(value)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, np.float32):
                            metrics_standard[key] = [float(v) for v in value]
                            break
            logger.info("Standard Test:\n%s", json.dumps(metrics_standard, indent=4)) # log result in a readable way
            with open(folder_path / "metric_standard.json", "w") as f:
                json.dump(metrics_standard, f, indent=2)

        if self.configs.save_arrays:
            for tensor_name in input_tensor_names:
                if array_dict[tensor_name] is not None:
                    np.save(folder_path / f"input_standard_{tensor_name}.npy", array_dict[tensor_name])
            for tensor_name in output_tensor_names:
                if array_dict[tensor_name] is not None:
                    np.save(folder_path / f"output_standard_{tensor_name}.npy", array_dict[tensor_name])

        return metrics_standard, array_dict

    
    
    def adaptive_test(self, adaptive_tester, test_loader, folder_path, iter_id=None):
        array_dict = {
            "x": [], "y": [], "x_mask": [], "y_mask": [], "sample_ID": [],
            "pred": [], "true": [], "mask": [], "uncertainty": [], "adaptation_triggered": []
        }
        
        total_adaptation_count = 0
        total_reliable_samples = 0
        total_unreliable_samples = 0
        total_samples = 0
        total_adaptation_loss = 0.0
    
        adaptive_threshold_history = []

        all_uncertainties = []
        reliable_uncertainties = []  
        unreliable_uncertainties = []  

        adaptive_tester.model.eval()

        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), leave=False, desc="Adaptive Test"):
            total_samples += batch["x"].shape[0]  
            
            device = accelerator.device if self.configs.use_multi_gpu else torch.device(f"cuda:{self.configs.gpu_id}")
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            outputs, uncertainty, adaptation_triggered, GDC_stats = adaptive_tester.process_batch(batch)
            
        

            if GDC_stats and 'reliable_samples' in GDC_stats:
                total_reliable_samples += GDC_stats['reliable_samples']
            else:
                pass
            if GDC_stats and 'unreliable_samples' in GDC_stats:
                total_unreliable_samples += GDC_stats['unreliable_samples']
            else:
                pass
            
            uncertainty_cpu = uncertainty.detach().cpu().numpy()
            all_uncertainties.extend(uncertainty_cpu)
            
            
            batch_all: list[dict] = accelerator.gather_for_metrics([batch])
            batch_all: dict = self._merge_gathered_dicts(batch_all)
            
            outputs_for_gather = {
                "pred": outputs["pred"],
                "true": outputs["true"], 
                "mask": outputs.get("mask", None),
                "uncertainty": uncertainty,
                "adaptation_triggered": torch.tensor([adaptation_triggered], device=device)
            }
            
            
            outputs_for_gather = self._prepare_for_gpu_gather(outputs_for_gather, device)
            outputs_all: list[dict] = accelerator.gather_for_metrics([outputs_for_gather])
            outputs_all: dict = self._merge_gathered_dicts(outputs_all)

            for tensor_name in ["x", "y", "x_mask", "y_mask", "sample_ID"]:
                if tensor_name in batch_all and batch_all[tensor_name] is not None:
                    array_dict[tensor_name].append(batch_all[tensor_name].detach().cpu().numpy())
            
            for tensor_name in ["pred", "true", "mask", "uncertainty", "adaptation_triggered"]:
                if tensor_name in outputs_all and outputs_all[tensor_name] is not None:
                    array_dict[tensor_name].append(outputs_all[tensor_name].detach().cpu().numpy())

        for key in array_dict:
            if len(array_dict[key]) > 0:
                array_dict[key] = np.concatenate(array_dict[key], axis=0)
            else:
                array_dict[key] = None

        

        metrics_adaptive = None
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            metric_dict = {k: v for k, v in array_dict.items() if k not in ["uncertainty", "adaptation_triggered"]}
            metrics_adaptive = metric(**metric_dict)
            
            if (self.configs.wandb and accelerator.is_main_process and self.configs.is_training) or self.configs.sweep:
                import wandb
                wandb.log({
                    "loss_test_adaptive": np.mean(metrics_adaptive["MSE"]),
                })
        
        # loss 
        avg_adapt_loss = total_adaptation_loss / total_adaptation_count if total_adaptation_count > 0 else 0.0
        
        GDC_summary = {
            'total_batches': len(test_loader),
            'total_samples': total_samples,
            'total_adaptation_count': total_adaptation_count,
            'total_reliable_samples': total_reliable_samples,
            'total_unreliable_samples': total_unreliable_samples,
            'reliable_ratio': total_reliable_samples / total_samples if total_samples > 0 else 0,
            'unreliable_ratio': total_unreliable_samples / total_samples if total_samples > 0 else 0,
            'average_adaptation_loss': avg_adapt_loss,
            'adaptation_steps': adaptive_tester.adaptation_steps,
            'adaptive_threshold_history': adaptive_threshold_history,
            'final_adaptive_threshold': adaptive_threshold_history[-1] if adaptive_threshold_history else None,
            'GDC_stats': adaptive_tester.GDC_stats
        }
        
       
        
        if metrics_adaptive is not None:
            for key, value in metrics_adaptive.items():
                if isinstance(value, np.float32):
                    metrics_adaptive[key] = float(value)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, np.float32):
                            metrics_adaptive[key] = [float(v) for v in value]
                            break
            
    
            
            adaptive_folder_path = folder_path / "adaptive_GDC_results"
            adaptive_folder_path.mkdir(exist_ok=True)
            
            with open(adaptive_folder_path / "metric_adaptive_GDC.json", "w") as f:
                json.dump(metrics_adaptive, f, indent=2)
            
            with open(adaptive_folder_path / "adaptive_GDC_stats.json", "w") as f:
                json.dump(GDC_summary, f, indent=2)
            
            GDC_cali = self._unwrap(adaptive_tester.adaptive_GDC_calibration)  
            expert_stats = GDC_cali.get_expert_stats()
            with open(adaptive_folder_path / "expert_stats.json", "w") as f:
                json.dump(expert_stats, f, indent=2)
            
            if self.configs.save_arrays:
                for tensor_name in array_dict:
                    if array_dict[tensor_name] is not None:
                        np.save(adaptive_folder_path / f"adaptive_GDC_{tensor_name}.npy", array_dict[tensor_name])
        
        return metrics_adaptive, array_dict


    def test(self) -> None:
        logger.info('>>>>>>> testing start <<<<<<<')
        
        # convert task_name to task_key for storage folder naming
        task_key_mapping = {
            "short_term_forecast": "forecasting",
            "long_term_forecast": "forecasting",
        }
        if self.configs.test_flop:
            self.configs.batch_size = 1
            logger.debug("batch_size automatically overwritten to 1.")
            test_params_flop(
                model=self._build_model().to(self.device), 
                x_shape=(self.configs.seq_len,self.configs.enc_in),
                model_id=self.configs.model_id,
                task_key=task_key_mapping[self.configs.task_name] if self.configs.task_name in task_key_mapping.keys() else self.configs.task_name
            )
            exit(0)

        if self.configs.test_train_time:
            self.configs.batch_size = 32
            logger.debug("batch_size automatically overwritten to 32.")
            train_data, train_loader = self._get_data(flag='train')
            test_train_time(
                model=self._build_model().to(self.device), 
                dataloader=train_loader,
                criterion=self._select_criterion(),
                model_id=self.configs.model_id,
                dataset_name=self.configs.dataset_name,
                gpu=self.configs.gpu_id,
                seq_len=self.configs.seq_len,
                pred_len=self.configs.pred_len,
                task_key=task_key_mapping[self.configs.task_name] if self.configs.task_name in task_key_mapping.keys() else self.configs.task_name,
                retain_graph=self.configs.retain_graph
            )
            exit(0)

        if self.configs.test_gpu_memory:
            self.configs.batch_size = 32
            logger.debug("batch_size automatically overwritten to 32.")
            train_data, train_loader = self._get_data(flag='train')
            batch = next(iter(train_loader))
            batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}
            model = self._build_model().to(self.device).train()
            test_gpu_memory(
                model=model,
                batch=batch,
                model_id=self.configs.model_id,
                dataset_name=self.configs.dataset_name,
                gpu=self.configs.gpu_id,
                seq_len=self.configs.seq_len,
                pred_len=self.configs.pred_len,
                task_key=task_key_mapping[self.configs.task_name] if self.configs.task_name in task_key_mapping.keys() else self.configs.task_name
            )
            exit(0)

        if self.configs.test_dataset_statistics:
            _, data_loader = self._get_data(flag='test_all')
            n_observations_raw = 0
            n_observations_all = 0
            logger.info(f"""Testing Dataset '{self.configs.dataset_name}':
            - seq_len={self.configs.seq_len}
            - pred_len={self.configs.pred_len}
            - batch_size={self.configs.batch_size}
            - collate_fn='{self.configs.collate_fn}'""")
            logger.warning("Make sure seq_len and pred_len are correctly set.")
            for batch in tqdm(data_loader):
                n_observations_raw += np.sum(batch["x_mask"].detach().cpu().numpy())
                n_observations_raw += np.sum(batch["y_mask"].detach().cpu().numpy())
                n_observations_all += np.sum(np.ones_like(batch["x_mask"].detach().cpu().numpy()))
                n_observations_all += np.sum(np.ones_like(batch["y_mask"].detach().cpu().numpy()))

            logger.info(f"No. observations (raw): {n_observations_raw}")
            logger.info(f"No. observations (all): {n_observations_all}")
            exit(0)



        # test_all will test the model on all available sets (train, val, test). Needs to be supported by the dataset
        flag = "test_all" if self.configs.test_all else "test"
        test_data, test_loader = self._get_data(flag=flag)

        # checkpoint_location will be used for both loading model weights and saving testing results
        checkpoint_location, checkpoint_location_itr = self._find_latest_checkpoint()
        if checkpoint_location is None:
            logger.error("No checkpoint found for testing. Please check --checkpoint_path and --checkpoints_test arguments.")
            return
        all_iter_folders = self._get_all_iteration_folders(checkpoint_location)
        actual_itrs = len(all_iter_folders) if all_iter_folders else 1
        logger.info(f"find {actual_itrs} iteration folders for testing")

        # check if uncertainty estimator needs to be trained for 
        if (self.configs.enable_ and 
            self.configs.enable_uncertainty_training and 
            not self._check_any_uncertainty_model_exists()):
            
            self.train_uncertainty_estimator_for_all_iters(checkpoint_location)

        # init global statistics storage for uncertainty analysis
        self.global_uncertainty_stats = {
            'all_uncertainties': [],
            'reliable_uncertainties': [],
            'unreliable_uncertainties': [],
            'iter_ids': [],
            'adaptive_threshold_history': []
        }
        iter_improvements: list[dict] = []

        # iterate through checkpoints for each iteration and perform testing
        for itr_i in range(actual_itrs):
            if self.configs.checkpoints_test is None:
                checkpoint_location_itr = checkpoint_location / f"iter{itr_i}"
            else:
                checkpoint_location_itr = Path(self.configs.checkpoints_test)

            self._setup_uncertainty_paths(checkpoint_location_itr, itr_i)

            model_test = self._build_model().eval()
            
            # if load_checkpoints_test is True, try to load model weights for testing.
            if self.configs.load_checkpoints_test:
                checkpoint_file = checkpoint_location_itr / "pytorch_model.bin"
                if checkpoint_file.exists():
                    try: 
                        # model state dict cannot be modified after accelerator.prepare
                        original_state_dict = self._get_state_dict(checkpoint_file)
                        load_result = model_test.load_state_dict(original_state_dict, strict=False)
                        if load_result.missing_keys or load_result.unexpected_keys:
                            logger.warning(f"""The following keys in checkpoint are not correctly loaded:
                            {load_result.missing_keys=}
                            {load_result.unexpected_keys=}

                            Results may be incorrect!
                            """)
                    except Exception as e:
                        logger.exception(f"{e}", stack_info=True)
                        logger.exception(f"Failed to load checkpoint file at {checkpoint_file}. Skipping it...")
                        continue
                else:
                    try:
                        # when weights are large (>10GB), they will be saved in several files
                        load_checkpoint_in_model(model_test, checkpoint_location_itr)
                    except Exception as e:
                        logger.exception(f"{e}", stack_info=True)
                        logger.exception(f"Failed to load checkpoint file at {checkpoint_file}. Skipping it...")
                        continue

            if self.configs.use_multi_gpu:
                model_test, test_loader = accelerator.prepare(model_test, test_loader)
                device_runtime = accelerator.device
            else:
                device_runtime = torch.device(f"cuda:{self.configs.gpu_id}" if self.configs.use_gpu else "cpu")
                model_test = model_test.to(device_runtime)

            subfolder_eval = f'eval_{datetime.datetime.now().strftime("%Y_%m%d_%H%M")}'
            folder_path = checkpoint_location_itr / subfolder_eval
            folder_path.mkdir(exist_ok=True)
            logger.info(f"Testing results will be saved under {folder_path}")
    
            # step1: standard test without Online Learning
            logger.info(f">>>>>>> {itr_i} step1: standard test <<<<<<<")
            metrics_standard, standard_array_dict = self.standard_test(model_test, test_loader, folder_path)

            # step2: adaptive GDC test with Online Learning, only if enabled in configs
            if self.configs.enable_:
                logger.info(f">>>>>>>{itr_i} step2: UnderCali <<<<<<<")
                
                # reload the model for adaptive test, to avoid potential influence of model state from standard test.
                model_test_adaptive = self._build_model().eval()
                if self.configs.load_checkpoints_test:
                    checkpoint_file = checkpoint_location_itr / "pytorch_model.bin"
                    if checkpoint_file.exists():
                        original_state_dict = self._get_state_dict(checkpoint_file)
                        model_test_adaptive.load_state_dict(original_state_dict, strict=False)
                
                model_test_adaptive, test_loader_adaptive = accelerator.prepare(model_test_adaptive, test_loader)
                if not self.configs.use_multi_gpu:
                    model_test_adaptive = model_test_adaptive.to(f"cuda:{self.configs.gpu_id}")
                
                # load uncertainty estimator for current iteration
                uncertainty_estimator = self.load_uncertainty_estimator_for_iter(checkpoint_location_itr, itr_i)
                
                if uncertainty_estimator is not None:
                    adaptive_tester = AdaptiveTester(
                        model=model_test_adaptive,
                        uncertainty_estimator=uncertainty_estimator,
                        configs=self.configs,
                        device=self.device
                    )
                
                    # adaptive test
                    metrics_adaptive, adaptive_array_dict = self.adaptive_test(
                        adaptive_tester, test_loader_adaptive, folder_path, iter_id=itr_i
                    )
                    
                    if metrics_standard is not None and metrics_adaptive is not None:
                        logger.info(f">>>>>>> {itr_i} test  <<<<<<<")
                        logger.info("standard test- MSE: %.6f, MAE: %.6f", 
                                metrics_standard.get('MSE', 0), metrics_standard.get('MAE', 0))
                        logger.info("adaptive test - MSE: %.6f, MAE: %.6f", 
                                metrics_adaptive.get('MSE', 0), metrics_adaptive.get('MAE', 0))
                        
                        mse_improvement = (metrics_standard.get('MSE', 0) - metrics_adaptive.get('MSE', 0)) / metrics_standard.get('MSE', 0) * 100
                        mae_improvement = (metrics_standard.get('MAE', 0) - metrics_adaptive.get('MAE', 0)) / metrics_standard.get('MAE', 0) * 100
                        
                        logger.info("MSE improvement: %.2f%%", mse_improvement)
                        logger.info("MAE improvement: %.2f%%", mae_improvement)

                        iter_improvements.append({
                            "iter": itr_i,
                            "standard_mse": float(metrics_standard.get('MSE', 0)),
                            "standard_mae": float(metrics_standard.get('MAE', 0)),
                            "adaptive_mse": float(metrics_adaptive.get('MSE', 0)),
                            "adaptive_mae": float(metrics_adaptive.get('MAE', 0)),
                            "mse_improvement_percent": float(mse_improvement),
                            "mae_improvement_percent": float(mae_improvement),
                        })
                        
                        comparison_results = {
                            "standard": metrics_standard,
                            "adaptive": metrics_adaptive,
                            "improvement": {
                                "mse_improvement_percent": mse_improvement,
                                "mae_improvement_percent": mae_improvement
                            }
                        }
                        with open(folder_path / "adaptive_GDC_comparison_results.json", "w") as f:
                            json.dump(comparison_results, f, indent=2)
                else:
                    logger.warning(f"cannot adaptive test for iter {itr_i}")

            logger.info(f'>>>>>>>{itr_i} testing finished <<<<<<<')

        logger.info('\>>>>>>> 所有迭代测试完成，开始对比分析（基于内存列表） <<<<<<<')

        # iter_improvements 
        if iter_improvements and len(iter_improvements) > 0:
            all_standard_results = {}
            all_adaptive_results = {}
            for item in iter_improvements:
                itr_i = item.get("iter", -1)
                all_standard_results[itr_i] = {
                    "MSE": item.get("standard_mse", 0.0),
                    "MAE": item.get("standard_mae", 0.0),
                }
                all_adaptive_results[itr_i] = {
                    "MSE": item.get("adaptive_mse", 0.0),
                    "MAE": item.get("adaptive_mae", 0.0),
                }

            self._print_comparison_summary(all_standard_results, all_adaptive_results)

            logger.info("\n================ adaptive test improvement summary ================")
            for item in iter_improvements:
                logger.info(
                    "iter %d: standard(MSE=%.4f, MAE=%.4f) | adaptive(MSE=%.4f, MAE=%.4f) | improve(MSE=%.2f%%, MAE=%.2f%%)",
                    item.get("iter", -1),
                    item.get("standard_mse", 0.0), item.get("standard_mae", 0.0),
                    item.get("adaptive_mse", 0.0), item.get("adaptive_mae", 0.0),
                    item.get("mse_improvement_percent", 0.0), item.get("mae_improvement_percent", 0.0)
                )
        else:
            logger.info("iter_improvements is NULL")

        logger.info('\>>>>>>> testing finished <<<<<<<')



    def _print_comparison_summary(self, standard_results, adaptive_results):
        if not standard_results:
            return
        
        print("\n" + "="*60)
        print("           Comparison of Adaptive Test Results Across All Iterations")
        print("="*60)
        
        std_mse = [m.get('MSE', 0) for m in standard_results.values()]
        std_mae = [m.get('MAE', 0) for m in standard_results.values()]
        avg_std_mse, std_std_mse = np.mean(std_mse), np.std(std_mse)
        avg_std_mae, std_std_mae = np.mean(std_mae), np.std(std_mae)
        
        print(f"standard test ({len(standard_results)}iterations):")
        print(f"  mean MSE: {avg_std_mse:.4f} ± {std_std_mse:.4f}")
        print(f"  mean MAE: {avg_std_mae:.4f} ± {std_std_mae:.4f}")
        
        if adaptive_results:
            ada_mse = [m.get('MSE', 0) for m in adaptive_results.values()]
            ada_mae = [m.get('MAE', 0) for m in adaptive_results.values()]
            avg_ada_mse, std_ada_mse = np.mean(ada_mse), np.std(ada_mse)
            avg_ada_mae, std_ada_mae = np.mean(ada_mae), np.std(ada_mae)
            
            print(f"\n adaptive test ({len(adaptive_results)}iterations):")
            print(f"  mean MSE: {avg_ada_mse:.4f} ± {std_ada_mse:.4f}")
            print(f"  mean MAE: {avg_ada_mae:.4f} ± {std_ada_mae:.4f}")
            
            mse_imp = (avg_std_mse - avg_ada_mse) / avg_std_mse * 100
            mae_imp = (avg_std_mae - avg_ada_mae) / avg_std_mae * 100
            
            print(f"\n improvement:")
            print(f"  MSE: {mse_imp:+.2f}%")
            print(f"  MAE: {mae_imp:+.2f}%")
        
        print("="*60)

    
