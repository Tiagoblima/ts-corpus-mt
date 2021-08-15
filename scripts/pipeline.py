import argparse
import datetime
import os
import re

import numpy as np
import pandas as pd
import torch
import wandb

wandb.login(key="8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4")

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--encoder', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

parser.add_argument('--epochs', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

# parser.add_argument('--src_corpus', metavar='N', type=str,
#                  help='an integer for the accumulator', required=True)
# parser.add_argument('--tgt_corpus', metavar='N', type=str,
#                  help='an integer for the accumulator', required=True)

parser.add_argument('--embedding', action='store_true',
                    help='an integer for the accumulator')

args = parser.parse_args()
TARGET_CORPUS = ['naa', 'nbv', 'nvi', 'nlth']

SOURCE_CORPUS = 'arc'
ENCODER = args.encoder
DATASET_DIR = '../datasets/'
N_STEPS = args.epochs


def create_folders(paths=None):
    if paths is None:
        paths = []

    for path in paths:
        try:
            os.makedirs(path)
        except OSError:
            pass


class Pipeline:

    def __init__(self, source, targets, corpus_weights):
        self.targets = targets
        self.source = source
        self.cps_weights = corpus_weights
        today = datetime.datetime.now()
        self.exp_id = today.strftime("%d_%m_%Y_%H:%M:%S")

        self.exp_path = f'../exps/exp_{self.exp_id}'
        self.report_path = os.path.join(self.exp_path, 'reports')
        self.pred_path = os.path.join(self.exp_path, 'prediction')
        create_folders([self.report_path, self.pred_path, os.path.join(self.exp_path, 'val')])

        self.config_path = os.path.join(self.exp_path, f'config.yaml')
        self.config_file = open(self.config_path, 'w')

    def select_dataset(self):

        tgt_val_texts = []

        val_porp = [w / sum(self.cps_weights) for w in self.cps_weights]
        src_val_path = os.path.join(DATASET_DIR, 'val', f'{self.source}-val.txt')
        src_val_texts = open(src_val_path, encoding='utf8').readlines()
        data_config = f"save_data: ../{self.exp_path}\n" \
                      f"src_vocab: ../{self.exp_path}/vocab/dataset.vocab\n" \
                      f"tgt_vocab: ../{self.exp_path}/vocab/dataset.vocab\n"
        for i, target in enumerate(self.targets):
            corpus_path = os.path.join(DATASET_DIR, 'train', f'corpus_{self.source}-{target}')
            tgt_val_path = os.path.join(DATASET_DIR, 'val', f'{target}-val.txt')
            tgt_val_text = open(tgt_val_path, encoding='utf8').readlines()

            tgt_val_texts.extend(tgt_val_text[:int(val_porp[i] * len(tgt_val_text))])

            source_path = os.path.join(corpus_path, f'{self.source}-train.txt')
            target_path = os.path.join(corpus_path, f'{target}-train.txt')

            data_config += f"data:                            \n" \
                           f"   corpus_{self.source}-{target}:\n" \
                           f"           path_src: {source_path}\n" \
                           f"           path_tgt: {target_path}\n" \
                           f"           weights:  {self.cps_weights[i]}\n"

            self.config_file.write(data_config)

        tgt_val_path = os.path.join(self.exp_path, 'val', 'tgt-val.txt')
        src_val_path = os.path.join(self.exp_path, 'val', 'src-val.txt')

        np.savetxt(src_val_path, src_val_texts, fmt='%s', encoding='utf-8')

        np.savetxt(tgt_val_path, tgt_val_texts, fmt='%s', encoding='utf-8')

        src_val_path = os.path.join(DATASET_DIR, 'val', f'{self.source}-val.txt')

        data_config = "   valid:\n" \
                      f"      path_src: {src_val_path}\n" \
                      f"      path_tgt: {tgt_val_path}\n"

        self.config_file.write(data_config)

    def add_embedding(self):
        if args.embedding:
            emb_config = "both_embeddings: ../glove_dir/glove_s300.txt\nembeddings_type: \"GloVe\""
            self.config_file.write(emb_config)

    def config_setup(self):
        model_config = open(f'../{ENCODER}/{ENCODER}.config.yaml').read()
        data_config = open(os.path.join(DATASET_DIR, 'data.config.yaml')).read()
        self.config_file.write("word_vec_size: 300\nrnn_size: 300\n")
        self.config_file.write(data_config)
        self.select_dataset()
        self.add_embedding()

        logs_path = os.path.join(self.exp_path, 'runs/fit')
        self.config_file.write(f"\ntensorboard_log_dir: {logs_path}")
        model_path = f"\nsave_model: {self.exp_path}/run/model\n"
        self.config_file.write(model_path)
        self.config_file.write(model_config)

        if torch.cuda.is_available():
            self.config_file.write(f"\nsave_checkpoint_steps: {N_STEPS}\ntrain_steps: {N_STEPS}")
            self.config_file.write('\ngpu_ranks: [0]\n')
            self.config_file.write("batch_size: 32\nvalid_batch_size: 32")
        else:
            self.config_file.write(f"\nsave_checkpoint_steps: {N_STEPS}\ntrain_steps: {N_STEPS}")
            self.config_file.write("\nbatch_size: 32\nvalid_batch_size: 32")
        self.config_file.close()

    def train(self):

        os.system(f'onmt_build_vocab -config {self.config_path} -n_sample 50000')
        wandb.init(project="ts-mt")
        os.system(f'onmt_train -config {self.config_path}')

    def translate(self):

        model_path = f'{self.exp_path}/run/model_step_{N_STEPS}.pt'
        test_file = f"{DATASET_DIR}/test/{self.source}-test.txt"

        pred_file = os.path.join(self.pred_path, f"{'_'.join(self.targets)}-pred.txt")

        translate_cmd = f'onmt_translate -model {model_path} -src {test_file} -output {pred_file} -verbose '
        if torch.cuda.is_available():
            translate_cmd += ' -gpu 0'
        os.system(translate_cmd)

    def evaluate(self):

        pred_file = os.path.join(self.pred_path, f"{'_'.join(self.targets)}-pred.txt")

        preds = open(pred_file, encoding='utf-8').readlines()

        inputs = open(
            os.path.join(DATASET_DIR, 'test', f'{self.source}-test.txt'),
            encoding='utf-8').readlines()

        result_dict = {
            'src_sent': inputs,
            'pred_sent': preds,
        }

        for i, version in enumerate(self.targets):
            ref_file = f'reference_{version}'
            target = open(os.path.join(DATASET_DIR, f'test/references', ref_file + '.txt'),
                          encoding='utf8').readlines()

            result_dict[version] = target

        df = pd.DataFrame(result_dict)

        # refs = df.loc[:, self.targets].to_numpy()

        # def list_bleu(tup):
        #     return sentence_bleu(sys_sent=tup[0], ref_sents=tup[1])

        # list_score = list(map(list_bleu, zip(preds, refs)))

        # df['bleu_score'] = list_score
        # refs = df.loc[:, self.targets].T.to_numpy()
        # bleu_score = corpus_bleu(refs_sents=refs, sys_sents=preds)
        # sari_score = corpus_sari(orig_sents=inputs, refs_sents=refs, sys_sents=preds)

        # result[SOURCE_CORPUS + '-' + '_'.join(self.targets)] = {
        #      'BLEU': round(bleu_score, 2),
        #     'SARI': round(sari_score, 2),
        # }
        if len(self.targets) == 1:
            tgts = '_'.join(self.targets)
        else:
            tgts = 'multicorpus'

        if args.embedding:
            filename = f'{self.source}-{tgts}.{ENCODER}.embedding.csv'
        else:
            filename = f'{self.source}-{tgts}.{ENCODER}.csv'
        df.to_csv(os.path.join(self.report_path, filename))
        # pd.DataFrame.from_dict(result, orient='index').to_csv(
        #   os.path.join(self.report_path, f'corpus_report.csv'))

    def run_pipeline(self):

        self.config_setup()
        self.train()
        self.translate()
        self.evaluate()


def main():
    for target in zip(TARGET_CORPUS):
        pipe = Pipeline(SOURCE_CORPUS, target, [1])
        pipe.run_pipeline()

    pipe = Pipeline(SOURCE_CORPUS, TARGET_CORPUS, [6, 7, 6, 8])
    pipe.run_pipeline()


if __name__ == '__main__':
    main()
