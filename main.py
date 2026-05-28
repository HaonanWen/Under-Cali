
import random
from pathlib import Path
import datetime
import importlib
import yaml
from dataclasses import asdict
import pprint
import os
# Set the PyTorch CUDA allocation config via Python environment variable instead of shell export
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import numpy as np

from exp.under_cali import Exp_Main
# from exp.exp_batch_new_ablation import Exp_Main

from utils.globals import logger, accelerator
from utils.configs import configs

hyperparameters_sweep: dict[str, dict[str, list]] = {}

'''
Implement three modes:

Intra-iteration training: Train the corresponding uncertainty estimator immediately after finishing the main model training for each iteration.
Batch training: After training the main models for all iterations, train the uncertainty estimators for all iterations in batch.
On-demand training: Automatically train the uncertainty estimator during testing if none is found.
'''

def set_global_seed(seed: int, deterministic: bool = True):
    import os, random, numpy as np, torch

    # Python / NumPy / Torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def main():
    # random seed
    fix_seed_list = range(1, 1 + configs.itr)

    configs.use_gpu = True if torch.cuda.is_available() and configs.use_gpu else False

    Exp = Exp_Main

    def start_exp_train() -> Exp_Main:
        # save training config file for reference
        path = Path(configs.checkpoints) / configs.dataset_name / configs.model_name / configs.model_id / f"{configs.seq_len}_{configs.pred_len}" / configs.subfolder_train / f"iter{configs.itr_i}" # same as the one in Exp_Main.train()
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Training iter{configs.itr_i} save to: {path}")
        with open(path / "configs.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(asdict(configs), f, default_flow_style=False)
        # init exp tracker
        if (configs.wandb and accelerator.is_main_process) or configs.sweep:
            import wandb
            run = wandb.init(
                # Set the project where this run will be logged
                project="YOUR_PROJECT_NAME",
                # Track hyperparameters and run metadata
                config={
                    "model_name": configs.model_name,
                    "model_id": configs.model_id,
                    "dataset_name": configs.dataset_name,
                    "seq_len": configs.seq_len,
                    "pred_len": configs.pred_len,
                    "learning_rate": configs.learning_rate,
                    "batch_size": configs.batch_size,
                    "enable_": configs.enable_,
                    "enable_uncertainty_training": configs.enable_uncertainty_training
                },
                dir=path
            )
            # overwrite model hyperparameters when sweeping
            for attribute_name in hyperparameters_sweep.keys():
                setattr(configs, attribute_name, getattr(wandb.config, attribute_name))

        accelerator.project_configuration.set_directories(project_dir=path)

        exp = Exp(configs)
        exp.train()
        return exp

    if configs.sweep:
        '''
        Currently, wandb sweep with huggingface accelerate multi GPU is tricky, use at your own risk.
        It is running N cases of hyperparameter settings at the same time, each case in a GPU. It's NOT running 1 case using all GPUs.
        - `wandb.sweep` is only created in the main process
        - `wandb.agent` is created in every process, where the sweep_id is obtained via tmp file on disk
        - `accelerate.backward` and `accelerate.log` are not used
        '''
        # hyperparameter search using wandb sweep
        logger.info('>>>>>>> sweeping start <<<<<<<')

        subfolder = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
        configs.subfolder_train = subfolder
        # Automatically enable wandb logging when sweeping
        configs.wandb = 1
        logger.debug('wandb=1: Weight & Bias logging is automatically enabled')

        # ignore itr, only train once for each combination
        configs.itr = 1
        configs.itr_i = 0
        logger.debug('itr=1: training iteration is automatically overwritten to 1')

        # random.seed(fix_seed_list[configs.itr_i])
        # torch.manual_seed(fix_seed_list[configs.itr_i])
        # np.random.seed(fix_seed_list[configs.itr_i])
        set_global_seed(fix_seed_list[configs.itr_i])


        exp = start_exp_train()  

        if configs.enable_uncertainty_training:
            exp.train_uncertainty_only_for_all_iters()

        exp.test()
        
    elif configs.is_training:
        '''
        Normal train&test
        '''
        subfolder = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
        configs.subfolder_train = subfolder

        exp_objects = []

        for i in range(configs.itr):
            configs.itr_i = i

            random.seed(fix_seed_list[i])
            torch.manual_seed(fix_seed_list[i])
            np.random.seed(fix_seed_list[i])

            exp = start_exp_train()
            exp_objects.append(exp)
           
            if getattr(configs, 'train_uncertainty_immediately', False) and getattr(configs, 'enable_uncertainty_training', False):
                exp.train_uncertainty_estimator_for_current_iter()
            
            torch.cuda.empty_cache()
        
        if (not getattr(configs, 'train_uncertainty_immediately', False) and 
            getattr(configs, 'enable_uncertainty_training', False)):
            exp_objects[-1].train_uncertainty_estimator_for_all_iters()
        
        exp_objects[-1].test()
    else:
        '''
        test only
        '''
        exp = Exp(configs)
        if (getattr(configs, 'enable_', False) and 
            getattr(configs, 'enable_uncertainty_training', False) and
            not exp._check_any_uncertainty_model_exists()):
            
            exp.train_uncertainty_estimator_for_all_iters()
        
        exp.test()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # warp the codes, such that errors will only be outputted from the main process
    try:
        if not configs.sweep:
            main()
        else:
            # first determine the hyperparameters actually accessed by model
            from utils.ExpConfigs import ExpConfigsTracker
            configs_tracker = ExpConfigsTracker(configs)
            model_module = importlib.import_module("models." + configs.model_name)
            model = model_module.Model(configs_tracker)
            del model
            accessed_configs: set[str] = configs_tracker.get_accessed_attributes()
            max_count = 1
            for accessed_config in accessed_configs:
                try:
                    ref_values = configs.get_sweep_values(accessed_config)
                    if ref_values and (type(ref_values) is list):
                        hyperparameters_sweep[accessed_config] = {
                            "values": ref_values
                        }
                        max_count *= len(ref_values)
                except Exception as e:
                    continue
            # grid search if <=16, otherwise bayes
            sweep_method = "grid" if max_count <= 16 else "bayes"
            max_count = min(max_count, 16)

            if hyperparameters_sweep == {}:
                logger.error(f"No hyperparameter to be searched, stopping now..")
                logger.debug(f"{configs.model_name} access these attributes in ExpConfigs:")
                configs_tracker.print_access_report()
                logger.debug("""Possible reasons: (1) The model does not access any hyperparameters in ExpConfigs; (2) The accessed hyperparameters have not set their metadata properly. Check the ExpConfigs class in utils/ExpConfigs.py. Example metadata setting:
                d_model: int = field(metadata={"sweep": [32, 64, 128, 256]})""")
                exit(0)
            else:
                logger.info(f"""{len(hyperparameters_sweep)} hyperparameters and {max_count} runs using "{sweep_method}" as the sweep method: \n{pprint.pformat(hyperparameters_sweep)}""")
                
            import wandb
            sweep_configuration = {
                "method": sweep_method,
                "metric": {"goal": "minimize", "name": "loss_val_best"},
                "parameters": hyperparameters_sweep
            }
            temp_file_path = "storage/tmp.txt"
            if accelerator.is_main_process:
                sweep_id = wandb.sweep(sweep=sweep_configuration, project="YOUR_PROJECT_NAME")
                with open(temp_file_path, mode='w', encoding="utf-8") as f:
                    f.write(sweep_id)
            accelerator.wait_for_everyone()
            sweep_id = None
            with open(temp_file_path, mode='r', encoding="utf-8") as f:
                sweep_id = f.readline()
            wandb.agent(
                sweep_id, 
                function=main, 
                project="YOUR_PROJECT_NAME",
                count=max_count
            )
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\nProcess interrupted...")
    except Exception as e:
        if accelerator.is_main_process:
            logger.exception(f"{e}", stack_info=True)
            exit(1)
