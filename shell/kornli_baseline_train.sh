activate() {
  /home/heerak/workspace/EFL_implementation/venv/bin/activate
}

cd /home/heerak/workspace/EFL_implementation

python src/train.py --train_file snli_1.0_train.ko.csv --validation_file xnli.dev.ko.csv \
--model_name_or_path pretrain/checkpoint-2000000 --vocab_path tokenizer/version_1.9 --output_dir finetuning/kornli \
--checkpointing_steps 1000 --pad_to_max_length --with_tracking --report_to wandb --entity raki-1203 --seed 42