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


def get_train_df():
    train_dfs = []
    for train_corpus in os.listdir(os.path.join(DATASET_DIR, 'train')):
        src_train_file = os.path.join(DATASET_DIR, train_corpus, f'src-train.txt')
        tgt_train_file = os.path.join(DATASET_DIR, train_corpus, f'tgt-train.txt')
        src_text = open(src_train_file).readlines()
        tgt_text = open(tgt_train_file).readlines()
        train_dfs.append(pd.DataFrame([src_text, tgt_text], columns=['input_text', 'target_text']))
    return pd.concat(train_dfs)


def get_val_df():
    src_val_file = os.path.join(DATASET_DIR, 'val', 'src-val.txt')
    tgt_val_file = os.path.join(DATASET_DIR, 'val', 'tgt-val.txt')
    src_text = open(src_val_file).readlines()
    tgt_text = open(tgt_val_file).readlines()

    return pd.DataFrame([src_text, tgt_text], columns=['input_text', 'target_text'])


def save_as_file(filename, df):
    dataset = df.loc[:, ['input_text', 'target_text']].to_numpy()
    dataset = dataset.reshape(dataset.shape[0] * 2, 1).squeeze()
    np.savetxt(filename, dataset, fmt='%s', encoding='utf8')


# model_args = {
#     "length_penalty": 0.001,
#     "reprocess_input_data": True,
#     "overwrite_output_dir": True,
#     "max_seq_length": 30,
#     "train_batch_size": 32,
#     "num_train_epochs": 20,
#     "save_eval_checkpoints": False,
#     "save_model_every_epoch": False,
#     "evaluate_generated_text": True,
#     "evaluate_during_training_verbose": True,
#     "use_multiprocessing": True,
#     "max_length": 30,
#     "manual_seed": 4,
#     "save_steps": 58300,
#     'wandb_project': pair,
#        "repetition_penalty": 100
# }
def fine_tuning(model_args):

    model = LanguageModelingModel("bert", "neuralmind/bert-base-portuguese-cased", args=model_args)

    train_file = os.path.join(DATASET_DIR, 'train.txt')
    eval_file = os.path.join(DATASET_DIR, 'val.txt')

    train_df = get_train_df()
    val_df = get_val_df()

    save_as_file(train_file, train_df)
    save_as_file(eval_file, val_df)

    model.train_model(train_file)
    result = model.eval_model(eval_file)
    print("Evaluation: ", result)


def train(model_args):
    model = Seq2SeqModel(
        "bert",
        "outputs",
        "outputs",
        args=model_args,
        use_cuda=torch.cuda.is_available(),
    )
    train_df = get_train_df()
    val_df = get_val_df()
    model.train_model(train_df)
    results = model.eval_model(val_df)
    print(f"Evaluation: {results}")

    pred_spt = model.predict(open(os.path.join(DATASET_DIR, 'test', 'src-test.txt')))
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

    fine_tuning(model_args)
    train(model_args)
