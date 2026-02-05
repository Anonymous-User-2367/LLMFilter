model_name=LLMFilter_FullLlama
dataset=pendulum
master_port=05098
num_process=4

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run.py \
  --task_name short_term_filtering \
  --is_training 1 \
  --root_path ./dataset/$dataset \
  --data_path $dataset.csv \
  --model_id $dataset \
  --model $model_name \
  --llm_ckp_dir model/llama-7b \
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
  --train_epochs 20 \
  --weight_decay 0.00001 \
  --use_multi_gpu \
  --save False \
