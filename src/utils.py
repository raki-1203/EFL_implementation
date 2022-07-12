import os
import pickle

import pandas as pd
from datasets import Features, Value, DatasetDict, Dataset

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
    train_path = os.path.join(data_dir, args.train_file)
    train_df = pd.read_csv(train_path)
    valid_path = os.path.join(data_dir, args.validation_file)
    valid_df = pd.read_csv(valid_path)
    test_path = os.path.join(data_dir, args.test_file)
    test_df = pd.read_csv(test_path)

    f = Features({'sentence1': Value(dtype='string', id=None),
                  'sentence2': Value(dtype='string', id=None),
                  'label': Value(dtype='string', id=None)})

    datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=f),
                            'validation': Dataset.from_pandas(valid_df, features=f),
                            'test': Dataset.from_pandas(test_df, features=f)})

    datasets.save_to_disk(os.path.join(data_dir, 'kornli_dataset'))
