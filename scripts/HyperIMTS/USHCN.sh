use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

model_name="$(basename "$(dirname "$(readlink -f "$0")")")" # folder name

dataset_root_path=storage/datasets/USHCN
model_id=$model_name
dataset_name=$(basename "$0" .sh) # file name

seq_len=150  # 50 ,100 150

#  grid
trg_alphas=(0.25)
trg_ks=(0.75)
all_alphas=(0.75)
all_ks=(0.25)

for pred_len in 3; do
    echo "Running  grid a_trg=$trg_alphas k_trg=$trg_ks a_all=$all_alphas k_all =$all_ks"
        $launch_command main_new.py \
        --is_training 0 \
        --loss "MSE" \
        --d_model 128 \
        --n_layers 1 \
        --n_heads 4 \
        --use_multi_gpu $use_multi_gpu \
        --dataset_root_path $dataset_root_path \
        --model_id $model_id \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 5 \
        --dec_in 5 \
        --c_out 5 \
        --patch_len 10 \
        --train_epochs 300 \
        --patience 10 \
        --val_interval 1 \
        --itr 5 \
        --batch_size 16 \
        --learning_rate 0.001 \
        --gpu_id 0 \
        --_ema_alpha_trigger "$trg_alphas" \
        --_std_k_trigger "$trg_ks" \
        --_ema_alpha_alloc "$all_alphas" \
        --_std_k_alloc "$all_ks" 
done

