import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

dataset_file = './database/train.csv'
test_file = './database/sample_submission.csv'

train_path = './datalist/train.txt'
val_path = './datalist/val.txt'
test_path = './datalist/test.txt'


def load_file(train_dir, test_dir):
    df = pd.read_csv(train_dir)
    df = shuffle(df, random_state=1)
    df['variety'] = pd.factorize(df['variety'])[0]

    idx_2_label = df['label'].unique()
    label_2_idx = {idx: label for (label, idx) in enumerate(idx_2_label)}
    df['path'] = 'train_images/' + df['label'] + '/' + df['image_id']

    train_df, val_df = train_test_split(df, test_size=0.2)
    # print(df['label'].value_counts())

    test_df = pd.read_csv(test_dir)
    test_df['path'] = 'test_images/' + test_df['image_id']

    return train_df, val_df, test_df, label_2_idx


def write_lines_2_train(path_dir, df, label_2_idx):
    paths = df['path'].values.tolist()
    labels = df['label'].values.tolist()
    with open(path_dir, 'w') as f:
        for i in range(len(paths)):
            f.write(paths[i] + ' ' + str(label_2_idx[labels[i]]) + '\n')
    print("done write to: ", path_dir)


def write_lines_2_test(path_dir, df):
    paths = df['path'].values.tolist()
    with open(path_dir, 'w') as f:
        for i in range(len(paths)):
            f.write(paths[i] + '\n')
    print("done write to: ", path_dir)


def main():
    train_list, val_list, test_list, l2i = load_file(dataset_file, test_file)

    # print(train_list['label'].value_counts())
    # print(val_list['label'].value_counts())

    write_lines_2_train(train_path, train_list, l2i)
    write_lines_2_train(val_path, val_list, l2i)
    write_lines_2_test(test_path, test_list)


if __name__ == '__main__':
    main()

