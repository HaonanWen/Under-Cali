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

seq_len=150   # 50,100, 150 


trg_alphas=(0.25)
trg_ks=(0.75)
all_alphas=(0.75)
all_ks=(0.25)


for pred_len in 3; do
    for a_trg in "${trg_alphas[@]}"; do
        for k_trg in "${trg_ks[@]}"; do
            for a_all in "${all_alphas[@]}"; do
                for k_all in "${all_ks[@]}"; do
                    echo "Running  grid a_trg=$a_trg k_trg=$k_trg a_all=$a_all k_all =$k_all"
                    $launch_command main.py \
                    --is_training 0 \
                    --d_model 64 \
                    --n_layers 1 \
                    --dropout 0.0 \
                    --node_dim 10 \
                    --collate_fn "collate_fn_patch" \
                    --patch_len 10 \
                    --loss "MSE" \
                    --n_heads 1 \
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
                    --train_epochs 300 \
                    --patience 10 \
                    --val_interval 1 \
                    --itr 5 \
                    --batch_size 16 \
                    --learning_rate 0.001 \
                    --_ema_alpha_trigger "$a_trg" \
                    --_std_k_trigger "$k_trg" \
                    --_ema_alpha_alloc "$a_all" \
                    --_std_k_alloc "$k_all"
                done
            done
        done
    done
done

