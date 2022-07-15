import os
import pickle

import pandas as pd
from datasets import Features, Value, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from transformers import AutoConfig, BertTokenizer, AutoTokenizer, RobertaForSequenceClassification, \
    AutoModelForSequenceClassification

project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, 'data')


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


def make_dataset(args):
    if args.train_file:
        train_path = os.path.join(data_dir, args.train_file)
    if args.validation_file:
        valid_path = os.path.join(data_dir, args.validation_file)
    if args.test_file:
        test_path = os.path.join(data_dir, args.test_file)

    if args.task_dataset == 'kornli':
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)

        f = Features({'sentence1': Value(dtype='string', id=None),
                      'sentence2': Value(dtype='string', id=None),
                      'label': Value(dtype='string', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=f),
                                'validation': Dataset.from_pandas(valid_df, features=f),
                                'test': Dataset.from_pandas(test_df, features=f)})

        datasets.save_to_disk(os.path.join(data_dir, 'kornli_dataset'))
    elif args.task_dataset == 'ksc':
        train_df = pd.read_csv(train_path)
        train_df.columns = ['sentence1', 'label']

        train_df, test_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'],
                                             shuffle=True, random_state=args.seed)

        train_df, valid_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'],
                                              shuffle=True, random_state=args.seed)

        f = Features({'sentence1': Value(dtype='string', id=None),
                      'label': Value(dtype='string', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=f),
                                'validation': Dataset.from_pandas(valid_df, features=f),
                                'test': Dataset.from_pandas(test_df, features=f)})

        datasets.save_to_disk(os.path.join(data_dir, 'ksc_dataset'))


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
