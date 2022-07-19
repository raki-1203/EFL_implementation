activate() {
  /home/heerak/workspace/EFL_implementation/venv/bin/activate
}

cd /home/heerak/workspace/EFL_implementation

#python src/train.py --train_file snli_1.0_train.ko.csv --validation_file xnli.dev.ko.csv --test_file xnli.test.ko.csv \
#--model_name_or_path pretrain/checkpoint-2000000 --vocab_path tokenizer/version_1.9 --output_dir finetuning/tbert-base/kornli/original \
#--checkpointing_steps 1000 --gradient_accumulation_steps 8 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-kornli-baseline --entity raki-1203 --seed 42 --learning_rate 5e-6

# cosine learning rate scheduler
#python src/train.py --train_file snli_1.0_train.ko.csv --validation_file xnli.dev.ko.csv --test_file xnli.test.ko.csv \
#--model_name_or_path pretrain/checkpoint-2000000 --vocab_path tokenizer/version_1.9 --output_dir finetuning/tbert-base/kornli/original \
#--checkpointing_steps 1000 --pad_to_max_length --with_tracking --report_to wandb \
#--run_name tbert-base-kornli-baseline --entity raki-1203 --seed 42 --learning_rate 5e-6 --lr_scheduler_type CosineAnnealingLR \
#--num_train_epochs 10

# entail / not entail 로 학습
python src/train.py --train_file xnli.train.ko.csv --validation_file xnli.dev.ko.csv --test_file xnli.test.ko.csv \
--model_name_or_path pretrain/checkpoint-2000000 --vocab_path tokenizer/version_1.9 --output_dir finetuning/tbert-base/kornli-efl/original \
--checkpointing_steps 1000 --pad_to_max_length --with_tracking --report_to wandb \
--run_name tbert-base-kornli-efl-original --entity raki-1203 --seed 42 --learning_rate 5e-6 --lr_scheduler_type ReduceLROnPlateau \
--num_train_epochs 10