activate() {
  /home/heerak/workspace/EFL_implementation/venv/bin/activate
}

cd /home/heerak/workspace/EFL_implementation

python efl_train.py --train_file snli_1.Korean_Singular_Conversation_Dataset.csv --task_dataset ksc \
--model_name_or_path klue/roberta-base --output_dir finetuning/klue-roberta-base/ksc/efl --num_train_epochs 10 \
--checkpointing_steps 1000 --gradient_accumulation_steps 1 --pad_to_max_length --with_tracking --report_to wandb \
--run_name roberta-base-ksc-efl --entity raki-1203 --seed 42 --learning_rate 1e-5 --lr_scheduler_type ReduceLROnPlateau
