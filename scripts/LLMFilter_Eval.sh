model_names=("LLMFilter_Llama")
datasets=("selkov")

for model_name in "${model_names[@]}"; do
    for dataset in "${datasets[@]}"; do
        # Extract the part before the underscore
        dataset_ckpt=${dataset%%_*}
        echo "Running model $model_name for dataset: $dataset"
        python -u run.py \
            --task_name short_term_filtering \
            --is_training 0 \
            --root_path ./dataset/$dataset \
            --data_path $dataset.csv \
            --model_id $dataset \
            --model $model_name \
            --llm_ckp_dir model/llama-7b \
            --checkpoints checkpoints/ \
            --window_length 40 \
            --data single_filtering \
            --batch_size 16 \
            --des 'Exp' \
            --itr 1 \
            --learning_rate 0.0001 \
            --loss 'MSE' \
            --use_amp \
            --gpu 1 \
            --mlp_hidden_dim 512 \
            --cosine \
            --tmax 10 \
            --train_epochs 10 \
            --weight_decay 0.00001 \
            --test_dir short_term_filtering_${dataset_ckpt}_${model_name}_single_filtering_wl40_lr0.0001_bt16_wd1e-05_hd512_hl2_cosTrue_mixFalse_Exp_0    
        echo "Completed run for dataset: $dataset with model: $model_name"
    done
done

