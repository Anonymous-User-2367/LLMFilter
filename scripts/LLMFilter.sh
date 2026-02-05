model_name=LLMFilter_Llama
dataset=hopf

python -u run.py \
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
  --save False \
  --mlp_hidden_dim 512 \
  --cosine \
  --tmax 10 \
  --train_epochs 10 \
  --weight_decay 0.00001 \

