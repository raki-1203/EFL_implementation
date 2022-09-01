activate() {
  /home/heerak/workspace/EFL_implementation/venv/bin/activate
}

cd /home/heerak/workspace/EFL_implementation

# TBERT 모델로 efl 진행
#python efl_train.py --train_file Korean_Singular_Conversation_Dataset.csv --task_dataset ksc \
#--model_name_or_path model/tbert_1.9 --vocab_path tokenizer/version_1.9 \
#--output_dir finetuning/tbert-base/ksc/efl --num_train_epochs 10 \
#--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-ksc-efl --entity raki-1203 --seed 42 --learning_rate 1e-5 --lr_scheduler_type ReduceLROnPlateau

# Description 2로 변경하고 epoch 5로 줄임
#python efl_train.py --train_file Korean_Singular_Conversation_Dataset.csv --task_dataset ksc \
#--model_name_or_path model/tbert_1.9 --vocab_path tokenizer/version_1.9 \
#--output_dir finetuning/tbert-base/ksc/efl-desc2 --num_train_epochs 5 \
#--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-ksc-efl-desc2 --entity raki-1203 --seed 42 --learning_rate 1e-5 --lr_scheduler_type ReduceLROnPlateau

# Description 3로 변경하고 kornli 학습한 tbert-base step_57000 사용하고 epoch 5로 줄임
#python efl_train.py --train_file Korean_Singular_Conversation_Dataset.csv --task_dataset ksc \
#--model_name_or_path finetuning/tbert-base/kornli-efl/original/step_57000 --vocab_path tokenizer/version_1.9 \
#--output_dir finetuning/tbert-base/ksc/efl-desc3 --num_train_epochs 5 \
#--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-ksc-efl-desc3 --entity raki-1203 --seed 42 --learning_rate 1e-5 --lr_scheduler_type ReduceLROnPlateau

# Description 1로 변경하고 kornli 학습한 tbert-base step_88000 사용하고 epoch 5로 줄임
#python efl_train.py --train_file Korean_Singular_Conversation_Dataset.csv --task_dataset ksc \
#--model_name_or_path finetuning/tbert-base/kornli-efl/original/step_88000 --vocab_path tokenizer/version_1.9 \
#--output_dir finetuning/tbert-base/ksc/efl-desc1 --num_train_epochs 5 \
#--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-ksc-efl-desc1 --entity raki-1203 --seed 42 --learning_rate 1e-5 \
#--lr_scheduler_type ReduceLROnPlateau --rdrop_coef 0.5

# Description 1로 변경하고 kornli 학습한 tbert-base step_102000 사용하고 epoch 10
#python efl_train.py --train_file Korean_Singular_Conversation_Dataset.csv --task_dataset ksc \
#--model_name_or_path finetuning/tbert-base/kornli-efl/patience_5/step_102000 --vocab_path tokenizer/version_1.9 \
#--output_dir finetuning/tbert-base/ksc/efl-desc1-rdrop_0.5-patience_5 --num_train_epochs 10 \
#--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-ksc-efl-desc1-rdrop_0.5-patience_5 --entity raki-1203 --seed 42 --learning_rate 1e-5 \
#--lr_scheduler_type ReduceLROnPlateau --rdrop_coef 0.5 --patience 5

# Description 1로 변경하고 kornli 학습한 tbert-base step_102000 사용하고 epoch 10 | rdrop 3
python efl_train.py --train_file Korean_Singular_Conversation_Dataset.csv --task_dataset ksc \
--model_name_or_path finetuning/tbert-base/kornli-efl/patience_5/step_102000 --vocab_path tokenizer/version_1.9 \
--output_dir finetuning/tbert-base/ksc/efl-desc1-rdrop_3-patience_5 --num_train_epochs 10 \
--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
--run_name tbert-base-ksc-efl-desc1-rdrop_3-patience_5 --entity raki-1203 --seed 42 --learning_rate 1e-5 \
--lr_scheduler_type ReduceLROnPlateau --rdrop_coef 3 --patience 5