import pandas as pd
import numpy as np
import os

from easse.sari import corpus_sari
from easse.bleu import corpus_bleu, sentence_bleu

REPORT_DIR = '../reports'


def save_final_report(base_dir='../new', out_dir='reports.csv'):
    result = {}
    for file in os.listdir(base_dir):
        df = pd.read_csv(os.path.join(base_dir, file))
        #  print(file.split('-')[0], round(df['sari_score'].to_numpy().mean(), 2))
        refs = np.expand_dims([df["trg_sent"].tolist()], axis=1)[0]

        sari_score = corpus_sari(orig_sents=df["src_sent"].tolist(),
                                 sys_sents=df["pred_sent"].tolist(),
                                 refs_sents=refs)

        bleu_score = corpus_bleu(
            sys_sents=df["pred_sent"].tolist(),
            refs_sents=refs
        )

        result[file.split('-')[0]] = {

            'SARI': round(sari_score, 2),

            'BLEU': round(bleu_score, 2),

        }
    pd.DataFrame.from_dict(result).T.to_csv(out_dir)


def main():
    save_final_report(base_dir='../new')
    score = corpus_sari(orig_sents=["About 95 species are currently accepted.", "Olá Mundo"],
                        sys_sents=["About 95 species are currently known.", "Oi Mundo"],
                        refs_sents=[["About 95 species are currently known.", "Olá Mundo"]])

    print(score)

    df = pd.read_csv('../reports/BERT-report#3.csv')
    indexes = df[df['bleu_score'] >= 0.9].index
    print(indexes)
    examples = []
    for report_file in os.listdir(REPORT_DIR):
        print(report_file.split('-')[0])
        report = pd.read_csv(os.path.join(REPORT_DIR, report_file))
        # for index in indexes:
        index = indexes[0]
        data = report.loc[index, ['pred_sent', 'src_sent', 'trg_sent', 'bleu_score']].T.to_dict()

        df = pd.DataFrame.from_dict(data, orient='index', columns=[report_file.split('-')[0]]).T
        examples.append(df)

    pd.concat(examples).to_csv('examples.csv')


if __name__ == '__main__':
    main()
