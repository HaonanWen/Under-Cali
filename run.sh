#!/bin/bash
mkdir -p logs
# for dataset_name in "ECL" "ETTh1" "ETTm1" "ILI" "Traffic" "Weather"; do
for dataset_name in  "HumanActivity" "P12" "USHCN" "MIMIC_III"; do
    # sh scripts/Ada_MSHyper/$dataset_name.sh >> logs/Ada_MSHyper_${dataset_name}.log 2>&1
    # sh scripts/Autoformer/$dataset_name.sh
    # sh scripts/BigST/$dataset_name.sh
    # sh scripts/Crossformer/$dataset_name.sh >> logs/Crossformer_${dataset_name}.log 2>&1
    # sh scripts/CRU/$dataset_name.sh >> logs/CRU_${dataset_name}.log 2>&1
    # sh scripts/DLinear/$dataset_name.sh
    # sh scripts/ETSformer/$dataset_name.sh
    # sh scripts/FEDformer/$dataset_name.sh
    # sh scripts/FiLM/$dataset_name.sh
    # sh scripts/FourierGNN/$dataset_name.sh
    # sh scripts/FreTS/$dataset_name.sh
    # sh scripts/GNeuralFlow/$dataset_name.sh >> logs/GNeuralFlow_${dataset_name}.log 2>&1
    # sh scripts/GraFITi/$dataset_name.sh >> logs/GraFITi_${dataset_name}.log 2>&1
    # sh scripts/GRU_D/$dataset_name.sh >> logs/GRU_D_${dataset_name}_ab.log 2>&1
    # bash scripts/Hi_Patch/$dataset_name.sh >> logs/Hi_Patch_${dataset_name}_ab.log 2>&1
    # sh scripts/higp/$dataset_name.sh
    # bash scripts/HyperIMTS/$dataset_name.sh >> logs/HyperIMTS_${dataset_name}.log 2>&1
    # sh scripts/Informer/$dataset_name.sh >> logs/Informer_${dataset_name}.log 2>&1
    # sh scripts/iTransformer/$dataset_name.sh >> logs/iTransforer__${dataset_name}.log 2>&1
    # sh scripts/Koopa/$dataset_name.sh
    # sh scripts/Latent_ODE/$dataset_name.sh
    # sh scripts/Leddam/$dataset_name.sh
    # sh scripts/LightTS/$dataset_name.sh
    # sh scripts/MambaSimple/$dataset_name.sh
    # sh scripts/MICN/$dataset_name.sh
    # sh scripts/MOIRAI/$dataset_name.sh
    # bash scripts/mTAN/$dataset_name.sh >> logs/mTAN_${dataset_name}_ab.log 2>&1
    # sh scripts/NeuralFlows/$dataset_name.sh
    # sh scripts/Nonstationary_Transformer/$dataset_name.sh
    # bash scripts/PatchTST/$dataset_name.sh >> logs/PatchTST_${dataset_name}.log 2>&1;
    # sh scripts/PrimeNet/$dataset_name.sh >> logs/PrimeNet_${dataset_name}_ab.log 2>&1;
    # sh scripts/Pyraformer/$dataset_name.sh;
    # bash scripts/Raindrop/$dataset_name.sh >> logs/Raindrop_${dataset_name}.log 2>&1;
    # sh scripts/Reformer/$dataset_name.sh >> logs/Reformer_${dataset_name}.log 2>&1;
    # sh scripts/SeFT/$dataset_name.sh >> logs/SeFT_${dataset_name}_ab.log 2>&1;
    # sh scripts/SegRNN/$dataset_name.sh;
    # sh scripts/TiDE/$dataset_name.sh;
    # sh scripts/TimeMixer/$dataset_name.sh;
    # sh scripts/TimesNet/$dataset_name.sh
    # bash scripts/tPatchGNN/$dataset_name.sh >> logs/tPatchGNN_${dataset_name}_ab.log 2>&1
    # sh scripts/Transformer/$dataset_name.sh
    # sh scripts/TSMixer/$dataset_name.sh
    # bash scripts/Warpformer/$dataset_name.sh >> logs/Warpformer_${dataset_name}_length12.log 2>&1
done
