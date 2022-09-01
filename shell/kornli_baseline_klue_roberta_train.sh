activate() {
  /home/heerak/workspace/EFL_implementation/venv/bin/activate
}

cd /home/heerak/workspace/EFL_implementation

python train.py --train_file snli_1.0_train.ko.csv --validation_file xnli.dev.ko.csv \
--test_file xnli.test.ko.csv --model_name_or_path klue/roberta-base --output_dir finetuning/klue-roberta-base/kornli/original \
--checkpointing_steps 1000 --gradient_accumulation_steps 8 --pad_to_max_length --with_tracking --report_to wandb \
--run_name roberta-base-kornli-baseline --entity raki-1203 --seed 42 --learning_rate 5e-6