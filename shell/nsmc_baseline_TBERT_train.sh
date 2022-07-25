activate() {
  /home/heerak/workspace/EFL_implementation/venv/bin/activate
}

cd /home/heerak/workspace/EFL_implementation

# TBERT NSMC original finetuning
python src/train.py --train_file ratings_train.csv --validation_file ratings_test.csv --task_dataset nsmc \
--model_name_or_path pretrain/checkpoint-2000000 --vocab_path tokenizer/version_1.9 \
--output_dir finetuning/tbert-base/nsmc/original --num_train_epochs 10 \
--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
--run_name tbert-base-nsmc-original --entity raki-1203 --seed 42 --learning_rate 1e-5 \
--lr_scheduler_type ReduceLROnPlateau --patience 5