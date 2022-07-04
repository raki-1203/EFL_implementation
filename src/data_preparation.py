import os

import pandas as pd


project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, 'data')


def make_csv_dataset(file_name):
    file_path = os.path.join(data_dir, file_name)
    extension = file_path.split('.')[-1]
    if extension not in ['tsv', 'xlsx']:
        raise ValueError('Only tsv or xlsx files can be used')
    if extension == 'tsv':
        df = pd.read_csv(file_path, sep='\t')
        df = df.rename(columns={'gold_label': 'label'})
    elif extension == 'xlsx':
        df = pd.read_excel(file_path, engine='openpyxl')
        df = df[['Sentence', 'Emotion']]
        df = df.rename(columns={'Sentence': 'sentence1', 'Emotion': 'label'})

    save_path = file_path[:-len(extension)] + 'csv'
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    make_csv_dataset(file_name='Korean_Singular_Conversation_Dataset.xlsx')
    make_csv_dataset(file_name='snli_1.0_train.ko.tsv')
    make_csv_dataset(file_name='xnli.dev.ko.tsv')
    make_csv_dataset(file_name='xnli.test.ko.tsv')
