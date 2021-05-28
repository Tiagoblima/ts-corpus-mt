import argparse
import os

import pandas as pd
from nltk.translate.bleu_score import corpus_bleu

REPORT_DIR = '../reports'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)
parser.add_argument('--embedding', action='store_true',
                    help='an integer for the accumulator')

args = parser.parse_args()


def save_final_report(base_dir='prediction', out_dir='reports.csv'):
    result = {}
    for file in os.listdir(base_dir):
        df = pd.read_csv(os.path.join(base_dir, file))
        #  print(file.split('-')[0], round(df['sari_score'].to_numpy().mean(), 2))
        refs = [[ref] for ref in df["trg_sent"].tolist()]

        bleu_score = corpus_bleu(refs, df["pred_sent"].tolist())

        result[file.split('.')[0]] = {
            'BLEU': round(bleu_score, 2),
        }
    pd.DataFrame.from_dict(result).T.to_csv(out_dir)


def main():
    encoder = args.model
    for file in os.listdir(encoder + '/prediction'):
        lang_pair = '-'.join(file.split('-')[:2])
        print(lang_pair)
        preds = open(os.path.join(encoder + '/prediction', file), encoding='utf-8').readlines()
        inputs = open(
            os.path.join('datasets/', lang_pair, 'test.' + lang_pair.split('-')[0]),
            encoding='utf-8').readlines()
        target = open(
            os.path.join('datasets/', lang_pair, 'test.' + lang_pair.split('-')[1]),
            encoding='utf-8').readlines()
        # report[lang_pair] =
        pd.DataFrame({
            'pred_sent': preds,
            'src_sent': inputs,
            'trg_sent': target
        }).to_csv(os.path.join(encoder + '/reports/', lang_pair + '.csv'))

    save_final_report(base_dir=encoder + '/reports', out_dir=encoder + '/reports/report.csv')


if __name__ == '__main__':
    main()
