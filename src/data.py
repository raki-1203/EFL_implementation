import random
import pandas as pd

from datasets import Features, Value, DatasetDict, Dataset


class KSCProcessor(object):
    """Processor for the Korean Singular Conversation dataset"""

    def __init__(self, negative_num=1):
        # Random negative sample number for efl strategy
        self.neg_num = negative_num

    def create_examples(self, datasets, task_label_description):
        """Creates examples for the training and dev sets."""

        train_datasets = datasets['train']
        valid_datasets = datasets['validation']
        test_datasets = datasets['test']

        train_examples = []
        valid_examples = []
        test_examples = []

        for example in train_datasets:
            true_label = example['label']
            neg_examples = []
            for label, label_description in task_label_description.items():
                new_example = dict()
                new_example['sentence1'] = example['sentence1']
                new_example['sentence2'] = label_description

                # Todo: handle imbalanced example, maybe hurt model performance
                if true_label == label:
                    new_example['label'] = 1
                    train_examples.append(new_example)
                else:
                    new_example['label'] = 0
                    neg_examples.append(new_example)
            neg_examples = random.sample(neg_examples, self.neg_num)
            train_examples.extend(neg_examples)
        for example in valid_datasets:
            true_label = str(example['label'])
            for label, label_description in task_label_description.items():
                new_example = dict()
                new_example['sentence1'] = example['sentence1']
                new_example['sentence2'] = label_description

                # Get true_label's index at task_label_description for evaluate
                true_label_index = list(task_label_description.keys()).index(true_label)
                new_example['label'] = true_label_index
                valid_examples.append(new_example)
        for example in test_datasets:
            true_label = str(example['label'])
            for label, label_description in task_label_description.items():
                new_example = dict()
                new_example['sentence1'] = example['sentence1']
                new_example['sentence2'] = label_description

                # Get true_label's index at task_label_description for evaluate
                true_label_index = list(task_label_description.keys()).index(true_label)
                new_example['label'] = true_label_index
                test_examples.append(new_example)

        train_examples = pd.DataFrame(train_examples)
        valid_examples = pd.DataFrame(valid_examples)
        test_examples = pd.DataFrame(test_examples)

        f = Features({'sentence1': Value(dtype='string', id=None),
                      'sentence2': Value(dtype='string', id=None),
                      'label': Value(dtype='int8', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(train_examples, features=f),
                                'validation': Dataset.from_pandas(valid_examples, features=f),
                                'test': Dataset.from_pandas(test_examples, features=f)})

        return datasets


class NSMCProcessor(object):
    """Processor for the Korean Singular Conversation dataset"""

    def __init__(self, negative_num=1):
        # Random negative sample number for efl strategy
        self.neg_num = negative_num

    def create_examples(self, datasets, task_label_description):
        """Creates examples for the training and dev sets."""

        train_datasets = datasets['train']
        valid_datasets = datasets['validation']

        train_examples = []
        valid_examples = []

        for example in train_datasets:
            true_label = example['label']
            neg_examples = []
            for label, label_description in task_label_description.items():
                new_example = dict()
                new_example['sentence1'] = example['sentence1']
                new_example['sentence2'] = label_description

                # Todo: handle imbalanced example, maybe hurt model performance
                if true_label == label:
                    new_example['label'] = 1
                    train_examples.append(new_example)
                else:
                    new_example['label'] = 0
                    neg_examples.append(new_example)
            neg_examples = random.sample(neg_examples, self.neg_num)
            train_examples.extend(neg_examples)
        for example in valid_datasets:
            true_label = str(example['label'])

            new_example = dict()
            new_example['sentence1'] = example['sentence1']
            new_example['sentence2'] = task_label_description['긍정']

            # Get true_label's index at task_label_description for evaluate
            true_label_index = list(task_label_description.keys()).index(true_label)
            new_example['label'] = true_label_index
            valid_examples.append(new_example)

            # for label, label_description in task_label_description.items():
            #     new_example = dict()
            #     new_example['sentence1'] = example['sentence1']
            #     new_example['sentence2'] = label_description
            #
            #     # Get true_label's index at task_label_description for evaluate
            #     true_label_index = list(task_label_description.keys()).index(true_label)
            #     new_example['label'] = true_label_index
            #     valid_examples.append(new_example)

        train_examples = pd.DataFrame(train_examples)
        valid_examples = pd.DataFrame(valid_examples)

        f = Features({'sentence1': Value(dtype='string', id=None),
                      'sentence2': Value(dtype='string', id=None),
                      'label': Value(dtype='int8', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(train_examples, features=f),
                                'validation': Dataset.from_pandas(valid_examples, features=f)})

        return datasets


class NaverShoppingProcessor(object):
    """Processor for the Korean Singular Conversation dataset"""

    def __init__(self, negative_num=1):
        # Random negative sample number for efl strategy
        self.neg_num = negative_num

    def create_examples(self, datasets, task_label_description):
        """Creates examples for the training and dev sets."""

        train_datasets = datasets['train']
        valid_datasets = datasets['validation']

        train_examples = []
        valid_examples = []

        for example in train_datasets:
            true_label = example['label']
            neg_examples = []
            for label, label_description in task_label_description.items():
                new_example = dict()
                new_example['sentence1'] = example['sentence1']
                new_example['sentence2'] = label_description

                # Todo: handle imbalanced example, maybe hurt model performance
                if true_label == label:
                    new_example['label'] = 1
                    train_examples.append(new_example)
                else:
                    new_example['label'] = 0
                    neg_examples.append(new_example)
            neg_examples = random.sample(neg_examples, self.neg_num)
            train_examples.extend(neg_examples)
        for example in valid_datasets:
            true_label = str(example['label'])

            new_example = dict()
            new_example['sentence1'] = example['sentence1']
            new_example['sentence2'] = task_label_description['부정']

            # Get true_label's index at task_label_description for evaluate
            true_label_index = list(task_label_description.keys()).index(true_label)
            new_example['label'] = 1 - true_label_index
            valid_examples.append(new_example)

        train_examples = pd.DataFrame(train_examples)
        valid_examples = pd.DataFrame(valid_examples)

        f = Features({'sentence1': Value(dtype='string', id=None),
                      'sentence2': Value(dtype='string', id=None),
                      'label': Value(dtype='int8', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(train_examples, features=f),
                                'validation': Dataset.from_pandas(valid_examples, features=f)})

        return datasets


processor_dict = {
    "ksc": KSCProcessor,
    "nsmc": NSMCProcessor,
    'naver_shopping': NaverShoppingProcessor,
}
