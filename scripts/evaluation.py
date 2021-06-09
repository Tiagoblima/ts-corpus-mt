import argparse
import os
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import pandas as pd
import sacrebleu
import torch
from nltk.tokenize import word_tokenize

refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
        ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
bleu = sacrebleu.corpus_bleu(sys, refs)
print(bleu.score)
REPORT_DIR = '../reports'


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--encoder', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)


parser.add_argument('--src_lang', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)
parser.add_argument('--tgt_lang', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

args = parser.parse_args()

ENCODER = args.encoder
DATASET_ROOT = '../datasets'


SOURCE_LANG = args.src_lang.lower()
TARGET_LANG = args.tgt_lang.lower()


def calculate_score(df):
    df_copy = df.copy()
    hypothesis = df_copy["hypothesis"].apply(lambda hyp: word_tokenize(hyp)).tolist()

    for i in range(3):
        df_copy[f"ref_{i + 1}"] = df_copy[f"ref_{i + 1}"].apply(lambda ref: word_tokenize(ref))
    references = df_copy.loc[:, [f'ref_{i + 1}' for i in range(3)]].to_numpy().tolist()
    print(hypothesis[:1])
    print(references[:1])
    print()
    bleu_score = corpus_bleu(references, hypothesis)

    df["bleu_score"] = list(map(lambda tup: round(sentence_bleu(tup[0], tup[1]), 2),
                                zip(references, hypothesis)))
    return df, bleu_score


def create_folders(paths=None):
    if paths is None:
        paths = []

    for path in paths:
        try:
            os.makedirs(path)
        except OSError:
            pass


def main():

    folder_name = '-'.join([SOURCE_LANG, TARGET_LANG])

    test_file = f"{DATASET_ROOT}/{folder_name}/test/{SOURCE_LANG}.txt"

    report_path = os.path.join('../' + ENCODER, "reports")
    pred_path = os.path.join('../' + ENCODER, "prediction")
    create_folders([report_path, pred_path])

    reference_files = f"{DATASET_ROOT}/{folder_name}/test/references"
    pred_file = os.path.join(pred_path, f"{SOURCE_LANG}-{TARGET_LANG}-pred.txt")

    hypothesis = open(pred_file, encoding='utf-8').readlines()

    sources = open(test_file, encoding='utf-8').readlines()
    df = pd.DataFrame({})
    df['hypothesis'] = hypothesis
    df['sources'] = sources

    for i, ref_file in enumerate(os.listdir(reference_files)):
        df[f"ref_{i + 1}"] = open(os.path.join(reference_files, ref_file), encoding='utf-8').readlines()
    df.to_csv(os.path.join(report_path, f"{folder_name}.report.csv"))

    df, bleu_score = calculate_score(df)
    print("Corpus Bleu score: ", bleu_score)

    report_file = open(os.path.join(report_path, f"{folder_name}.report.txt"), "w")
    report_file.write(f"BLEU SCORE\n")
    report_file.write(f"{folder_name}: {bleu_score}")
    report_file.close()
    df.to_csv(os.path.join(report_path, f"{folder_name}.score_report.csv"))


if __name__ == '__main__':
    main()
