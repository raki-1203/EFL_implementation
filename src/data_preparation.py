import os

import pandas as pd


project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, 'data')


def make_csv_dataset(file_name, file_name2=None):
    file_path = os.path.join(data_dir, file_name)
    extension = file_path.split('.')[-1]
    if file_name2 is not None:
        file_path2 = os.path.join(data_dir, file_name2)
    if extension not in ['tsv', 'xlsx', 'txt']:
        raise ValueError('Only tsv or xlsx or txt files can be used')
    if extension == 'tsv':
        sentence1, sentence2, label = [], [], []
        if file_name2 is not None:
            with open(file_path2, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    line = line.strip()
                    s1, s2, l = line.split('\t')
                    sentence1.append(s1)
                    sentence2.append(s2)
                    label.append(l)

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                line = line.strip()
                s1, s2, l = line.split('\t')
                sentence1.append(s1)
                sentence2.append(s2)
                label.append(l)

        df = pd.DataFrame()
        df['sentence1'] = sentence1
        df['sentence2'] = sentence2
        df['label'] = label
        print(f'{file_name}\noriginal data shape -> {df.shape}')
        df.dropna(inplace=True)
        print(f'data shape after NaN data drop -> {df.shape}')
    elif extension == 'xlsx':
        df = pd.read_excel(file_path, engine='openpyxl')
        df = df[['Sentence', 'Emotion']]
        df = df.rename(columns={'Sentence': 'sentence1', 'Emotion': 'label'})
    elif extension == 'txt':
        df = pd.read_csv(file_path, sep='\t')
        df = df[['document', 'label']]
        df.columns = ['sentence1', 'label']
        df['label'] = df['label'].apply(lambda x: '긍정' if x == 1 else '부정')

    if file_name2 is not None:
        save_path = 'xnli.train.ko.csv'
    else:
        save_path = file_path[:-len(extension)] + 'csv'
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    make_csv_dataset(file_name='Korean_Singular_Conversation_Dataset.xlsx')
    make_csv_dataset(file_name='snli_1.0_train.ko.tsv', file_name2='multinli.train.ko.tsv')
    make_csv_dataset(file_name='xnli.dev.ko.tsv')
    make_csv_dataset(file_name='xnli.test.ko.tsv')
    make_csv_dataset(file_name='ratings_train.txt')
    make_csv_dataset(file_name='ratings_test.txt')
