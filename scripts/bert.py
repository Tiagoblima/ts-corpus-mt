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


def get_datasets(pair):
    pair = 'arc-naa'
    train_df = pd.read_csv(os.path.join(DATASET_DIR, f'{pair}/train.csv')).dropna()
    val_df = pd.read_csv(os.path.join(DATASET_DIR, f'{pair}/val.csv')).dropna()
    test_df = pd.read_csv(os.path.join(DATASET_DIR, f'{pair}/test.csv')).dropna()
    return train_df, val_df, test_df


def save_as_file(filename, df):
    dataset = df.loc[:, ['input_text', 'target_text']].to_numpy()
    dataset = dataset.reshape(dataset.shape[0] * 2, 1).squeeze()
    np.savetxt(os.path.join(DATASET_DIR, filename), dataset, fmt='%s', encoding='utf8')


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
def fine_tuning(pair, model_args):
    model = LanguageModelingModel("bert", "neuralmind/bert-base-portuguese-cased", args=model_args)
    train_df, val_df, test_df = get_datasets(pair)
    train_file = 'train.txt'
    eval_file = 'val.txt'
    save_as_file(train_file, train_df)
    save_as_file(eval_file, train_df)
    model.train_model(train_file)
    result = model.eval_model(eval_file)
    print("Evaluation: ", result)


def train(pair, model_args):
    model = Seq2SeqModel(
        "bert",
        "outputs",
        "outputs",
        args=model_args,
        use_cuda=torch.cuda.is_available(),
    )
    train_df, val_df, test_df = get_datasets(pair)
    model.train_model(train_df)
    results = model.eval_model(val_df)
    print(f"Evaluation: {results}")
    pred_spt = model.predict(test_df['input_text'].tolist())
    pre_path = os.path.join(MODEL_DIR, 'prediction')
    try:
        os.makedirs(pre_path)
    except OSError:
        pass

    np.savetxt(os.path.join(pre_path, f'{pair}-pred.txt'), pred_spt, fmt="%s")


def main():
    for folder in os.listdir(DATASET_DIR):
        with open(os.path.join(MODEL_DIR, 'bert.config.json')) as json_file:
            model_args = json.load(json_file)
            model_args['wandb_project'] = folder
        fine_tuning(folder, model_args)
        train(folder, model_args)
