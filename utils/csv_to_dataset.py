import os
import pandas as pd

from datasets import Features, Value, DatasetDict, Dataset
from sklearn.model_selection import train_test_split

project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, 'data')


def make_dataset(args):
    if args.train_file:
        train_path = os.path.join(data_dir, args.train_file)
        train_df = pd.read_csv(train_path)
    if args.validation_file:
        valid_path = os.path.join(data_dir, args.validation_file)
        valid_df = pd.read_csv(valid_path)
    if args.test_file:
        test_path = os.path.join(data_dir, args.test_file)
        test_df = pd.read_csv(test_path)

    if args.task_dataset == 'kornli':
        f = Features({'sentence1': Value(dtype='string', id=None),
                      'sentence2': Value(dtype='string', id=None),
                      'label': Value(dtype='string', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=f),
                                'validation': Dataset.from_pandas(valid_df, features=f),
                                'test': Dataset.from_pandas(test_df, features=f)})

        datasets.save_to_disk(os.path.join(data_dir, 'kornli_dataset'))
    elif args.task_dataset == 'kornli-efl':
        train_df['label'] = train_df['label'].apply(lambda x: 'not entail' if x != 'entailment' else 'entail')
        valid_df['label'] = valid_df['label'].apply(lambda x: 'not entail' if x != 'entailment' else 'entail')
        test_df['label'] = test_df['label'].apply(lambda x: 'not entail' if x != 'entailment' else 'entail')

        f = Features({'sentence1': Value(dtype='string', id=None),
                      'sentence2': Value(dtype='string', id=None),
                      'label': Value(dtype='string', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=f),
                                'validation': Dataset.from_pandas(valid_df, features=f),
                                'test': Dataset.from_pandas(test_df, features=f)})

        datasets.save_to_disk(os.path.join(data_dir, 'kornli-efl_dataset'))
    elif args.task_dataset == 'ksc':
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
    elif args.task_dataset == 'nsmc':
        f = Features({'sentence1': Value(dtype='string', id=None),
                      'label': Value(dtype='string', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=f),
                                'validation': Dataset.from_pandas(valid_df, features=f)})

        datasets.save_to_disk(os.path.join(data_dir, 'nsmc_dataset'))
    elif args.task_dataset == 'naver_shopping':
        train_df, valid_df = train_test_split(train_df, test_size=0.3, stratify=train_df['label'],
                                              shuffle=True, random_state=args.seed)
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)

        f = Features({'sentence1': Value(dtype='string', id=None),
                      'label': Value(dtype='string', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=f),
                                'validation': Dataset.from_pandas(valid_df, features=f)})

        datasets.save_to_disk(os.path.join(data_dir, 'naver_shopping_dataset'))
