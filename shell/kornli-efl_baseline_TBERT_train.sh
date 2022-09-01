activate() {
  /home/heerak/workspace/EFL_implementation/venv/bin/activate
}

cd /home/heerak/workspace/EFL_implementation

# entail / not entail 로 학습
#python train.py --train_file xnli.train.ko.csv --validation_file xnli.dev.ko.csv --test_file xnli.test.ko.csv \
#--model_name_or_path model/tbert_1.9 --vocab_path tokenizer/version_1.9 --output_dir finetuning/tbert-base/kornli-efl/original \
#--checkpointing_steps 1000 --pad_to_max_length --with_tracking --report_to wandb --task_dataset kornli-efl \
#--run_name tbert-base-kornli-efl-original --entity raki-1203 --seed 42 --learning_rate 5e-6 --lr_scheduler_type ReduceLROnPlateau \
#--num_train_epochs 10

# entail / not entail 학습 | ReduceLROnPlateau patience 5
python train.py --train_file xnli.train.ko.csv --validation_file xnli.dev.ko.csv --test_file xnli.test.ko.csv \
--model_name_or_path model/tbert_1.9 --vocab_path tokenizer/version_1.9 --output_dir finetuning/tbert-base/kornli-efl/patience_5 \
--checkpointing_steps 1000 --pad_to_max_length --with_tracking --report_to wandb --task_dataset kornli-efl \
--run_name tbert-base-kornli-efl-patience_5 --entity raki-1203 --seed 42 --learning_rate 5e-6 --lr_scheduler_type ReduceLROnPlateau \
--num_train_epochs 10 --patience 5