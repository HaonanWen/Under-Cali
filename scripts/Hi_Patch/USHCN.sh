use_multi_gpu=1
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

model_name="$(basename "$(dirname "$(readlink -f "$0")")")" # folder name

dataset_root_path=storage/datasets/USHCN
model_id=$model_name
dataset_name=$(basename "$0" .sh) # file name

trg_alphas=(0.25)
trg_ks=(0.75)
all_alphas=(0.75)
all_ks=(0.25)

seq_len=60   #50, 100, 150
for pred_len in 3; do
    echo "Running UnderCali grid a_trg=$trg_alphas k_trg=$trg_ks a_all=$all_alphas k_all =$all_ks"
    $launch_command main.py \
        --is_training 1 \
        --collate_fn "collate_fn_patch" \
        --loss "MSE" \
        --n_heads 1 \
        --n_layers 1 \
        --d_model 64 \
        --patch_len 6 \
        --patch_stride 6 \
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
        --patience 5 \
        --val_interval 1 \
        --itr 5 \
        --batch_size 16 \
        --learning_rate 0.001
done

