# EFL_implementation

---

이 Repository 는 EFL 방식의 학습 구현 코드입니다.

비교를 위한 모델로 **klue/roberta-base** 와 **TBERT** 를 사용했습니다.

classification 을 위한 기본 fine-tuning 방식과 EFL 방법을 비교했습니다.

추가로 R-Drop 구현되어 있는 코드입니다.

## Dependency

- `python 3.9`
- `pip install -r requirements.txt`


## Data Preprocessing

참고 코드

- `data_preparation.py`
- `utils/data_preprocessor.py`
- `utils/task_label_description.py`

## R-Drop Loss

참고 코드

- `utils/loss.py`

## How to use

기본 fine-tuning 방식 학습

```
python train.py 
--train_file ratings_train.csv 
--validation_file ratings_test.csv 
--task_dataset nsmc 
--model_name_or_path {pretrained model path or hugingface model} 
--vocab_path {pretrained tokenizer_vocab} 
--output_dir {output model save path}
-num_train_epochs {epochs}
--checkpointing_steps {checkpoint steps} 
--gradient_accumulation_steps {gradient_accumulation_steps} 
--pad_to_max_length 
--with_tracking 
--report_to wandb 
--run_name {wandb run_name} 
--entity {wandb entity name}
--seed 42 
--learning_rate {learning_rate}
--lr_scheduler_type ReduceLROnPlateau 
--patience 5
```

EFL 방식 학습

```
python efl_train.py 
--train_file ratings_train.csv 
--validation_file ratings_test.csv 
--task_dataset nsmc 
--model_name_or_path {pretrained model path or hugingface model} 
--vocab_path {pretrained tokenizer_vocab} 
--output_dir {output model save path} 
--num_train_epochs {epochs}
--checkpointing_steps {checkpoint steps}
--gradient_accumulation_steps {gradient_accumulation_steps}
--pad_to_max_length 
--with_tracking 
--report_to wandb
--run_name {wandb run_name} 
--entity {wandb entity name}
--seed 42 
--learning_rate {learning_rate}
--lr_scheduler_type ReduceLROnPlateau 
--patience 5
```

EFL + R-Drop 방식 학습

```
python efl_train.py 
--train_file ratings_train.csv 
--validation_file ratings_test.csv 
--task_dataset nsmc 
--model_name_or_path {pretrained model path or hugingface model} 
--vocab_path {pretrained tokenizer_vocab} 
--output_dir {output model save path} 
--num_train_epochs {epochs}
--checkpointing_steps {checkpoint steps}
--gradient_accumulation_steps {gradient_accumulation_steps}
--pad_to_max_length 
--with_tracking 
--report_to wandb
--run_name {wandb run_name} 
--entity {wandb entity name}
--seed 42 
--learning_rate {learning_rate}
--lr_scheduler_type ReduceLROnPlateau 
--rdrop_coef 5
```




