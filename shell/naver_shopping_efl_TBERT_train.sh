activate() {
  /home/heerak_son/workspace/EFL_implementation/venv/bin/activate
}

cd /home/heerak_son/workspace/EFL_implementation

# Description 1 | tbert-base 기본 모델 | epoch 20 | rdrop 3
python efl_train.py --train_file naver_shopping.csv --task_dataset naver_shopping \
--model_name_or_path model/tbert_1.9 --vocab_path tokenizer/version_1.9 \
--output_dir finetuning/tbert-base/naver_shopping/efl-desc1-rdrop_3-patience_5 --num_train_epochs 20 \
--checkpointing_steps 1000 --gradient_accumulation_steps 2 --pad_to_max_length --with_tracking --report_to wandb \
--run_name tbert-base-naver_shopping-efl-desc1-rdrop_3-patience_5 --entity raki-1203 --seed 42 --learning_rate 1e-5 \
--lr_scheduler_type ReduceLROnPlateau --patience 5 --rdrop_coef 3 --per_device_train_batch_size 8