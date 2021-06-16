import json
import os
from simpletransformers.language_modeling import (
    LanguageModelingModel,
)
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import logging
import torch
import pandas as pd
import numpy as np
import wandb
from easse.bleu import sentence_bleu, corpus_bleu

from easse.sari import corpus_sari

DATASET_DIR = '../datasets'
MODEL_DIR = '../bert'

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
encoder_type = "bert"

wandb.login(key="8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4")

TARGET_CORPUS = ['nbv', 'nlth', 'nvi', 'naa']
SOURCE_CORPUS = 'arc'


def save_as_file(filename, df):
    dataset = df.loc[:, ['input_text', 'target_text']].to_numpy()
    dataset = dataset.reshape(dataset.shape[0] * 2, 1).squeeze()
    np.savetxt(filename, dataset, fmt='%s', encoding='utf8')


class Pipeline:

    def __init__(self, source_cps, tgt_corpus):
        self.target_cps = tgt_corpus
        self.source_cps = source_cps

        with open(os.path.join(MODEL_DIR, 'bert.config.json')) as json_file:
            model_args = json.load(json_file)

        model_args['wandb_project'] = "ts-mt"
        seq2seq_epochs = model_args['num_train_epochs']
        # fine_tunig epochs
        model_args['num_train_epochs'] = int(model_args['num_train_epochs'] / 2)
        self.bert_model = LanguageModelingModel("bert", "neuralmind/bert-base-portuguese-cased", args=model_args)
        model_args['num_train_epochs'] = seq2seq_epochs
        self.model_args = model_args
        self.train_dir = '../bert/'
        self.pre_path = os.path.join(self.train_dir, 'prediction', SOURCE_CORPUS + '-' + '_'.join(self.target_cps))
        try:
            os.makedirs(self.pre_path)
        except OSError:
            pass
        try:
            os.makedirs(self.train_dir)
        except OSError:
            pass
        self.val_file = self.train_dir + 'val.txt'
        self.train_file = self.train_dir + 'train.txt'

    def get_train_df(self):
        train_dfs = []
        for tgt_cps in self.target_cps:
            train_corpus = 'corpus_' + SOURCE_CORPUS + '-' + tgt_cps
            src_train_file = os.path.join(DATASET_DIR, 'train', train_corpus, f'{SOURCE_CORPUS}-train.txt')
            tgt_train_file = os.path.join(DATASET_DIR, 'train', train_corpus, f'{tgt_cps}-train.txt')
            src_text = open(src_train_file).readlines()
            tgt_text = open(tgt_train_file).readlines()
            train_dfs.append(pd.DataFrame({
                'input_text': src_text,
                'target_text': tgt_text
            }))

        return pd.concat(train_dfs)

    def get_val_df(self):
        val_dfs = []
        for tgt_cps in self.target_cps:
            src_val_file = os.path.join(DATASET_DIR, 'val', f'{SOURCE_CORPUS}-val.txt')
            tgt_val_file = os.path.join(DATASET_DIR, 'val', f'{tgt_cps}-val.txt')
            src_text = open(src_val_file).readlines()
            tgt_text = open(tgt_val_file).readlines()
            val_dfs.append(pd.DataFrame({
                'input_text': src_text,
                'target_text': tgt_text
            }))
        df = pd.concat(val_dfs)

        return df

    def fine_tuning(self):

        train_df = self.get_train_df()
        val_df = self.get_val_df()
        save_as_file(self.train_file, train_df)
        save_as_file(self.val_file, val_df)
        self.bert_model.train_model(self.train_file)
        result = self.bert_model.eval_model(self.val_file)
        print("Evaluation: ", result)

    def train_seq2seq(self):

        self.seq2seq_model = Seq2SeqModel(
            "bert",
            "outputs",
            "outputs",
            args=self.model_args,
            use_cuda=torch.cuda.is_available(),
        )
        train_df = self.get_train_df()
        val_df = self.get_val_df()
        self.seq2seq_model.train_model(train_df)
        results = self.seq2seq_model.eval_model(val_df)
        print(f"Evaluation: {results}")

    def translate(self):

        self.seq2seq_model = Seq2SeqModel(
            "bert",
            "outputs/encoder",
            "outputs/decoder",
            args=self.model_args,
            use_cuda=torch.cuda.is_available(),
        )

        self.preds = self.seq2seq_model.predict(
            open(os.path.join(DATASET_DIR, 'test', f'{SOURCE_CORPUS}-test.txt')).readlines())

        np.savetxt(os.path.join(self.pre_path, f'prediction.txt'), self.preds, fmt="%s")

    def evaluation(self):
        inputs = open(
            os.path.join(DATASET_DIR, 'test', f'{SOURCE_CORPUS}-test.txt'),
            encoding='utf-8').readlines()

        result_dict = {
            'src_sent': inputs,
            'pred_sent': self.preds,
        }

        for i, version in enumerate(self.target_cps):
            ref_file = f'reference_{version}'
            target = open(os.path.join(DATASET_DIR, f'test/references', ref_file + '.txt'),
                          encoding='utf8').readlines()

            result_dict[version] = target

        df = pd.DataFrame(result_dict)

        refs = df.loc[:, self.target_cps].to_numpy()

        def list_bleu(tup):
            return sentence_bleu(sys_sent=tup[0], ref_sents=tup[1])

        list_score = list(map(list_bleu, zip(self.preds, refs)))

        df['bleu_score'] = list_score
        refs = df.loc[:, self.target_cps].T.to_numpy()
        bleu_score = corpus_bleu(refs_sents=refs, sys_sents=self.preds)
        sari_score = corpus_sari(orig_sents=inputs, refs_sents=refs, sys_sents=self.preds)
        result = {SOURCE_CORPUS + '-' + '_'.join(self.target_cps): {
            'BLEU': round(bleu_score, 2),
            'SARI': round(sari_score, 2),
        }}

        df.to_csv(os.path.join(self.pre_path, 'sent_report.csv'))
        pd.DataFrame.from_dict(result, orient='index').to_csv(
            os.path.join(self.pre_path, f'corpus_report.csv'))


def main():
    for tgt_cps in TARGET_CORPUS:
        pipe = Pipeline(SOURCE_CORPUS, [tgt_cps])
        pipe.fine_tuning()
        pipe.train_seq2seq()
        pipe.translate()
        pipe.evaluation()


main()
