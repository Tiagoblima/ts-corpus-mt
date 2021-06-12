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

DATASET_DIR = '../datasets'
MODEL_DIR = '../bert'

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
encoder_type = "bert"

wandb.login(key="8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4")

TARGET_CORPUS = ['nbv', 'nlth', 'nvi', 'naa']
SOURCE_CORPUS = 'arc'


def get_train_df(tgt_corpus):
    train_dfs = []
    for tgt_cps in tgt_corpus:
        train_corpus = 'corpus_'+SOURCE_CORPUS + '-' + tgt_cps
        src_train_file = os.path.join(DATASET_DIR, 'train', train_corpus, f'{SOURCE_CORPUS}-train.txt')
        tgt_train_file = os.path.join(DATASET_DIR, 'train', train_corpus, f'{tgt_cps}-train.txt')
        src_text = open(src_train_file).readlines()
        tgt_text = open(tgt_train_file).readlines()
        train_dfs.append(pd.DataFrame({
            'input_text': src_text,
            'target_text': tgt_text
        }))
    return pd.concat(train_dfs)


def get_val_df(tgt_corpus):

    val_dfs = []
    for tgt_cps in tgt_corpus:

        src_val_file = os.path.join(DATASET_DIR, 'val', f'{SOURCE_CORPUS}-val.txt')
        tgt_val_file = os.path.join(DATASET_DIR, 'val', f'{tgt_cps}-val.txt')
        src_text = open(src_val_file).readlines()
        tgt_text = open(tgt_val_file).readlines()
        val_dfs.append(pd.DataFrame({
            'input_text': src_text,
            'target_text': tgt_text
        }))
    return pd.concat(val_dfs)


def save_as_file(filename, df):
    dataset = df.loc[:, ['input_text', 'target_text']].to_numpy()
    dataset = dataset.reshape(dataset.shape[0] * 2, 1).squeeze()
    np.savetxt(filename, dataset, fmt='%s', encoding='utf8')


def fine_tuning(model_args, tgt_cps):
    model = LanguageModelingModel("bert", "neuralmind/bert-base-portuguese-cased", args=model_args)

    train_file = os.path.join(DATASET_DIR, 'train.txt')
    eval_file = os.path.join(DATASET_DIR, 'val.txt')

    train_df = get_train_df(tgt_cps)
    val_df = get_val_df(tgt_cps)

    save_as_file(train_file, train_df)
    save_as_file(eval_file, val_df)

    model.train_model(train_file)
    result = model.eval_model(eval_file)
    print("Evaluation: ", result)


def train(model_args, tgt_cps):

    model = Seq2SeqModel(
        "bert",
        "outputs",
        "outputs",
        args=model_args,
        use_cuda=torch.cuda.is_available(),
    )
    train_df = get_train_df(tgt_cps)
    val_df = get_val_df(tgt_cps)
    model.train_model(train_df)
    results = model.eval_model(val_df)
    print(f"Evaluation: {results}")

    pred_spt = model.predict(open(os.path.join(DATASET_DIR, 'test', f'{SOURCE_CORPUS}-test.txt')).readlines())
    pre_path = os.path.join(MODEL_DIR, 'prediction')
    try:
        os.makedirs(pre_path)
    except OSError:
        pass

    np.savetxt(os.path.join(pre_path, f'prediction.txt'), pred_spt, fmt="%s")


def main():
    with open(os.path.join(MODEL_DIR, 'bert.config.json')) as json_file:
        model_args = json.load(json_file)
        model_args['wandb_project'] = "ts-mt"

    for tgt_cps in TARGET_CORPUS:
        fine_tuning(model_args, [tgt_cps])
        train(model_args, [tgt_cps])


main()
