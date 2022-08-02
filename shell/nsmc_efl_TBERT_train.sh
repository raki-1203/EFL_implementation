activate() {
  /home/heerak/workspace/EFL_implementation/venv/bin/activate
}

cd /home/heerak/workspace/EFL_implementation

# Description 1로 변경하고 kornli 학습한 tbert-base step_102000 사용하고 epoch 20 | rdrop 3
#python src/efl_train.py --train_file ratings_train.csv --validation_file ratings_test.csv --task_dataset nsmc \
#--model_name_or_path finetuning/tbert-base/kornli-efl/patience_5/step_102000 \
#--output_dir finetuning/tbert-base/nsmc/konli-efl-desc1-rdrop_3-patience_5 --num_train_epochs 20 \
#--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-nsmc-konli-efl-desc1-rdrop_3-patience_5 --entity raki-1203 --seed 42 --learning_rate 1e-5 \
#--lr_scheduler_type ReduceLROnPlateau --rdrop_coef 3 --patience 5

# Description 1로 변경하고 tbert-base 기본 모델 사용하고 epoch 20
#python src/efl_train.py --train_file ratings_train.csv --validation_file ratings_test.csv --task_dataset nsmc \
#--model_name_or_path pretrain/checkpoint-2000000 --vocab_path tokenizer/version_1.9 \
#--output_dir finetuning/tbert-base/nsmc/efl-desc1-patience_5 --num_train_epochs 20 \
#--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-nsmc-efl-desc1-patience_5 --entity raki-1203 --seed 42 --learning_rate 1e-5 \
#--lr_scheduler_type ReduceLROnPlateau --patience 5

# Description 1로 변경하고 tbert-base 기본 모델 사용하고 epoch 20 | rdrop 3
#python src/efl_train.py --train_file ratings_train.csv --validation_file ratings_test.csv --task_dataset nsmc \
#--model_name_or_path pretrain/checkpoint-2000000 --vocab_path tokenizer/version_1.9 \
#--output_dir finetuning/tbert-base/nsmc/efl-desc1-rdrop_3-patience_5 --num_train_epochs 20 \
#--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-nsmc-efl-desc1-rdrop_3-patience_5 --entity raki-1203 --seed 42 --learning_rate 1e-5 \
#--lr_scheduler_type ReduceLROnPlateau --patience 5 --rdrop_coef 3

# Description 1로 변경하고 tbert-base 기본 모델 사용하고 epoch 20 | rdrop 3 | checkpoint 부터 시
python src/efl_train.py --train_file ratings_train.csv --validation_file ratings_test.csv --task_dataset nsmc \
--model_name_or_path finetuning/tbert-base/nsmc/efl-desc1-rdrop_3-patience_5/step_162000 \
--output_dir finetuning/tbert-base/nsmc/efl-desc1-rdrop_3-patience_5 --num_train_epochs 20 \
--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
--run_name tbert-base-nsmc-efl-desc1-rdrop_3-patience_5 --entity raki-1203 --seed 42 --learning_rate 1e-5 \
--lr_scheduler_type ReduceLROnPlateau --patience 5 --rdrop_coef 3 --resume_from_checkpoint True
