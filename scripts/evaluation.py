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


DATASET_DIR = '../datasets'


def main():
    encoder = args.model
    result = {}
    model_dir = os.path.join('..', encoder)
    for file in os.listdir(os.path.join(model_dir, 'prediction')):

        preds = open(os.path.join(model_dir, 'prediction', file), encoding='utf-8').readlines()

        inputs = open(
            os.path.join(DATASET_DIR, 'src-test.txt'),
            encoding='utf-8').readlines()

        result_dict = {

            'src_sent': inputs,
            'pred_sent': preds,

        }
        refs = []
        reference_names = []
        for i, ref_file in enumerate(os.listdir(os.path.join(DATASET_DIR, 'test/references'))):
            target = open(os.path.join(DATASET_DIR, 'test/references', ref_file))
            reference_names.append(ref_file.split('.')[0])
            result_dict[ref_file.split('.')[0]] = target
            refs.append([[ref] for ref in target])

        list_bleu = lambda tup: sentence_bleu(sys_sent=tup[0], ref_sents=tup[1])
        list_score = list(map(list_bleu, zip(preds, refs)))
        print(len(preds), len(inputs), len(list_score))
        result_dict['bleu_score'] = list_score
        df = pd.DataFrame(result_dict)

        refs = np.expand_dims([df.loc[:, reference_names].tolist()], axis=1)[0]
        bleu_score = corpus_bleu(refs_sents=refs, sys_sents=preds)
        sari_score = corpus_sari(orig_sents=inputs, refs_sents=refs, sys_sents=preds)

        result["result"] = {
            'BLEU': round(bleu_score, 2),
            'SARI': round(sari_score, 2),
        }

        pd.DataFrame.from_dict(result, orient='index').to_csv(os.path.join(model_dir, 'reports', 'final_report.csv'))


if __name__ == '__main__':
    main()
