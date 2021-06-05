import argparse
import os
import numpy as np
import pandas as pd
from easse.sari import corpus_sari
from easse.bleu import corpus_bleu, sentence_bleu

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

    pd.DataFrame.from_dict(result).T.to_csv(out_dir)


def main():
    encoder = args.model
    result = {}
    model_dir = os.path.join('..', encoder)
    for file in os.listdir(os.path.join(model_dir, 'prediction')):
        lang_pair = '-'.join(file.split('-')[:2])
        print(lang_pair)

        preds = open(os.path.join(model_dir, 'prediction', file), encoding='utf-8').readlines()

        inputs = open(
            os.path.join('../datasets/', lang_pair, 'test.' + lang_pair.split('-')[0]),
            encoding='utf-8').readlines()
        target = open(
            os.path.join('../datasets/', lang_pair, 'test.' + lang_pair.split('-')[1]),
            encoding='utf-8').readlines()
        refs = [[ref] for ref in target]
        list_bleu = lambda tup: sentence_bleu(sys_sent=tup[0], ref_sents=tup[1])
        list_score = list(map(list_bleu, zip(preds, refs)))
        print(len(preds), len(inputs), len(list_score))
        df = pd.DataFrame({

            'src_sent': inputs,
            'pred_sent': preds,
            'trg_sent': target,
            'bleu_score': list_score
        })
        refs = np.expand_dims([df["trg_sent"].tolist()], axis=1)[0]
        bleu_score = corpus_bleu(refs_sents=refs, sys_sents=preds)
        sari_score = corpus_sari(orig_sents=inputs, refs_sents=refs, sys_sents=preds)

        result[lang_pair] = {
            'BLEU': round(bleu_score, 2),
            'SARI': round(sari_score, 2),
        }

        df.to_csv(os.path.join(model_dir, 'reports', lang_pair + '.csv'))

    pd.DataFrame.from_dict(result, orient='index').to_csv(os.path.join(model_dir, 'reports', 'final_report.csv'))


if __name__ == '__main__':
    main()
