import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--src_corpus', metavar='N', type=str,
                    help='an integer for the accumulator', required=False)

parser.add_argument('--tgt_corpus', metavar='N', type=str,
                    help='an integer for the accumulator', required=False)

parser.add_argument('--src_lang', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)
parser.add_argument('--tgt_lang', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

args = parser.parse_args()

DATASET_ROOT = '../datasets'

if not args.tgt_corpus:
    TARGET_FILES = ['naa', 'nbv', 'nvi', 'nlth']
else:
    TARGET_FILES = [args.tgt_corpus]

if not args.src_corpus:
    SOURCE_FILES = ["arc"]
else:
    SOURCE_FILES = [args.src_corpus]

DATAFRAME_FILE = '../ARC_NAA_NBV_NLTH_NVI_aligned.csv'

dataset = pd.read_csv(DATAFRAME_FILE).dropna()
train_df, eval_test_df = train_test_split(dataset, test_size=0.40)
eval_df, test_df = train_test_split(eval_test_df, test_size=0.30)

try:
    os.mkdir(DATASET_ROOT)
except OSError:
    pass

SOURCE_LANG = args.src_lang
TARGET_LANG = args.tgt_lang

readme_file_path = os.path.join(DATASET_ROOT, 'README.txt')
dataset_name = '-'.join([SOURCE_LANG, TARGET_LANG]).lower()
report_file = open(readme_file_path, 'w')
report_file.write('Dataset,Train, Validation, Test\n')
report_ = f"{dataset_name}, {train_df.shape[0]},  {eval_df.shape[0]}, {test_df.shape[0]}"
report_file.write(report_ + '\n')
report_file.close()

##################################################


folder_name = '-'.join([SOURCE_LANG, TARGET_LANG]).lower()

config_file_path = os.path.join(DATASET_ROOT, folder_name)

try:
    os.makedirs(config_file_path)
except OSError:
    pass
config_file_path = os.path.join(config_file_path, 'data.config.yaml')

config_file = open(config_file_path, 'w')

path_to_save = "save_data: " + os.path.join(DATASET_ROOT, f"{folder_name}/samples")
config_file.write(path_to_save + "\n")
source_path = "src_vocab: " + os.path.join(DATASET_ROOT, f"{folder_name}/vocab/{SOURCE_LANG}.vocab")
config_file.write(source_path + "\n")
tgt_path = f"tgt_vocab: " + os.path.join(DATASET_ROOT, f"{folder_name}/vocab/{TARGET_LANG}.vocab")
config_file.write(tgt_path + "\n")


def save_train_files(df):
    config_file.write("data:\n")

    for source in SOURCE_FILES:
        source_text = df[source].apply(str.strip)

        for i, target in enumerate(TARGET_FILES):

            corpus_path = os.path.join(DATASET_ROOT,
                                       '-'.join([SOURCE_LANG, TARGET_LANG]).lower(),
                                       'train', f'corpus_{target}')
            try:
                os.makedirs(corpus_path)
            except OSError:
                pass

            source_path = os.path.join(corpus_path, f'{SOURCE_LANG}.txt')
            target_path = os.path.join(corpus_path, f'{TARGET_LANG}.txt')

            data_config = f"   corpus_{i + 1}:\n" \
                          f"         path_src: {source_path}\n" \
                          f"         path_tgt: {target_path}\n"

            config_file.write(data_config)
            source_text.to_csv(source_path, header=None, index=None, sep=' ', mode='w')
            target_text = df[target].apply(str.strip)
            target_text.to_csv(target_path, header=None, index=None, sep=' ', mode='w')


save_train_files(train_df)


def save_val_files(df):
    source_path = ""
    target_path = ""
    for source in SOURCE_FILES:
        source_text = df[source].apply(str.strip)

        for i, target in enumerate(TARGET_FILES):

            corpus_path = os.path.join(DATASET_ROOT,
                                       '-'.join([SOURCE_LANG, TARGET_LANG]).lower(),
                                       'val')
            try:
                os.makedirs(corpus_path)
            except OSError:
                pass

            source_path = os.path.join(corpus_path, f'{SOURCE_LANG}.txt')
            source_text.to_csv(source_path, header=None, index=None, sep=' ', mode='w')

            target_text = df[target].apply(str.strip)
            target_path = os.path.join(corpus_path, f'{TARGET_LANG}.txt')
            target_text.to_csv(target_path, header=None, index=None, sep=' ', mode='w')

    data_config = "   valid:\n" \
                  f"      path_src: {source_path}\n" \
                  f"      path_tgt: {target_path}\n"
    config_file.write(data_config)


save_val_files(eval_df)


def save_test_files(df):
    for source in SOURCE_FILES:
        source_text = df[source].apply(str.strip)

        for i, target in enumerate(TARGET_FILES):

            corpus_path = os.path.join(DATASET_ROOT,
                                       '-'.join([SOURCE_LANG, TARGET_LANG]).lower(),
                                       'test')
            reference_path = os.path.join(corpus_path, 'references')

            try:
                os.makedirs(reference_path)
            except OSError:
                pass

            source_path = os.path.join(corpus_path, f'{SOURCE_LANG}.txt')
            source_text.to_csv(source_path, header=None, index=None, sep=' ', mode='w')

            target_text = df[target].apply(str.strip)
            target_path = os.path.join(reference_path, f'reference_{i + 1}.txt')
            target_text.to_csv(target_path, header=None, index=None, sep=' ', mode='w')


save_test_files(test_df)
config_file.close()
