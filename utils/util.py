import os
import pickle
import random
import numpy as np

import torch

from functools import partial
from konlpy.tag import Mecab
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, BertTokenizer, AutoTokenizer, RobertaForSequenceClassification, \
    AutoModelForSequenceClassification, get_linear_schedule_with_warmup, default_data_collator, DataCollatorWithPadding

project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, 'data')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_pickle(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ReduceLROnPlateauPatch(ReduceLROnPlateau, _LRScheduler):
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def get_config_tokenizer_model(args, num_labels):
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    if args.vocab_path is not None and os.path.isdir(args.vocab_path):
        tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                                  do_lower_case=False,
                                                  unk_token='<unk>',
                                                  sep_token='</s>',
                                                  pad_token='<pad>',
                                                  cls_token='<s>',
                                                  mask_token='<mask>',
                                                  model_max_length=args.max_length)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if os.path.isdir(args.model_name_or_path):
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )

    return config, tokenizer, model


def get_optimizer_grouped_parameters(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]

    return optimizer_grouped_parameters


def get_lr_scheduler(args, optimizer):
    if args.lr_scheduler_type == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=args.num_warmup_steps,
                                                       num_training_steps=args.max_train_steps,
                                                       )
    elif args.lr_scheduler_type == 'ReduceLROnPlateau':
        lr_scheduler = ReduceLROnPlateauPatch(optimizer, 'max', patience=args.patience, factor=0.9)
    elif args.lr_scheduler_type == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)

    return lr_scheduler


def get_dataloader(args, datasets, tokenizer, padding, label_to_id=None):
    # Preprocessing the datasets
    non_label_column_names = [name for name in datasets['train'].column_names if name != 'label']
    if 'sentence1' in non_label_column_names and 'sentence2' in non_label_column_names:
        sentence1_key, sentence2_key = 'sentence1', 'sentence2'
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done or max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if args.use_fp16 else None))

    preprocess_func = partial(preprocess_function,
                              args=args,
                              sentence_key=[sentence1_key, sentence2_key],
                              tokenizer=tokenizer,
                              padding=padding,
                              label_to_id=label_to_id,
                              )

    dataloader_list = []
    train_valid_test_list = ['train', 'validation', 'test']
    if len(datasets.keys()) == 2:
        train_valid_test_list = ['train', 'validation']
    for k in train_valid_test_list:
        k_dataset = datasets[k]

        k_dataset = k_dataset.map(
            preprocess_func,
            batched=True,
            remove_columns=datasets['train'].column_names,
            desc='Running tokenizer on dataset',
        )

        if k == 'train':
            k_dataloader = DataLoader(k_dataset,
                                      collate_fn=data_collator,
                                      batch_size=args.per_device_train_batch_size,
                                      shuffle=True,
                                      )
        else:
            k_dataloader = DataLoader(k_dataset,
                                      collate_fn=data_collator,
                                      batch_size=args.per_device_eval_batch_size,
                                      )

        dataloader_list.append(k_dataloader)

    return dataloader_list


def preprocess_function(examples, args, sentence_key, tokenizer, padding, label_to_id=None):
    sentence1_key, sentence2_key = sentence_key

    if args.vocab_path is not None:
        # TBERT 사용시 Mecab + Wordpiece Tokenizer 이기 때문에 Mecab 미리 적용 필요
        mecab = Mecab()
        examples[sentence1_key] = [' '.join(mecab.morphs(sentence)) for sentence in examples[sentence1_key]]

    # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*texts,
                       padding=padding,
                       max_length=args.max_length,
                       truncation=True,
                       return_token_type_ids=False,
                       )

    if 'label' in examples:
        # Map labels to IDs
        if label_to_id is None:
            result['labels'] = examples['label']
        else:
            result['labels'] = [label_to_id[l] for l in examples['label']]

    return result
