import argparse
import os

import nltk
import pandas as pd
import torch
import wandb
from easse.bleu import sentence_bleu, corpus_bleu
from easse.sari import corpus_sari

wandb.login(key="8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4")

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--encoder', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

parser.add_argument('--epochs', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

parser.add_argument('--src_corpus', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

parser.add_argument('--tgt_corpus', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

parser.add_argument('--embedding', action='store_true',
                    help='an integer for the accumulator')

args = parser.parse_args()

nltk.download('punkt')
ENCODER = args.encoder
ROOT_DIR = f'../{ENCODER}'
training_steps = args.epochs
DATASET_DIR = '../datasets/'

SOURCE_FILES = args.src_corpus.split(',')
TARGET_FILES = args.tgt_corpus.split(',')


def select_dataset(config_file):
    for source in SOURCE_FILES:

        for i, target in enumerate(TARGET_FILES):
            corpus_path = os.path.join(DATASET_DIR, 'train', f'corpus_{source}-{target}')
            try:
                os.makedirs(corpus_path)
            except OSError:
                pass

            try:
                os.makedirs(corpus_path)
            except OSError:
                pass

            source_path = os.path.join(corpus_path, f'{source}-train.txt')
            target_path = os.path.join(corpus_path, f'{target}-train.txt')

            data_config = f"   corpus_{source}-{target}:\n" \
                          f"           path_src: {source_path}\n" \
                          f"           path_tgt: {target_path}\n"
            config_file.write(data_config)

    source_path = os.path.join(DATASET_DIR, 'val', f'{args.src_corpus}-val.txt')
    target_path = os.path.join(DATASET_DIR, 'val', f'{args.tgt_corpus}-val.txt')
    data_config = "   valid:\n" \
                  f"      path_src: {source_path}\n" \
                  f"      path_tgt: {target_path}\n"
    config_file.write(data_config)


def create_config_file():
    global training_steps

    model_config = open(f'../{ENCODER}/{ENCODER}.config.yaml').read()
    data_config = open(os.path.join(DATASET_DIR, 'data.config.yaml')).read()

    if args.embedding:
        emb_config = "both_embeddings: ../glove_dir/glove_s300.txt\nembeddings_type: \"GloVe\"\nword_vec_size: 300\n\n"
        model_config += emb_config

    config_file_path = os.path.join('../', ENCODER, 'config_files', f'{ENCODER}.yaml')

    file = open(config_file_path, 'w')

    file.write(data_config)
    select_dataset(file)
    logs_path = os.path.join(ROOT_DIR, 'runs/fit')
    file.write(f"tensorboard_log_dir: {logs_path}\n")

    model_path = f"save_model: ../{ENCODER}/run/model\n"
    file.write(model_path)

    file.write(model_config)

    if torch.cuda.is_available():
        file.write(f"\nsave_checkpoint_steps: {training_steps}\ntrain_steps: {training_steps}")
        file.write('\ngpu_ranks: [0]\n')
        file.write("batch_size: 32\nvalid_batch_size: 32")
    else:
        file.write(f"\nsave_checkpoint_steps: {training_steps}\ntrain_steps: {training_steps}")
        file.write("\nbatch_size: 32\nvalid_batch_size: 32")
    file.close()
    return config_file_path


def create_folders(paths=None):
    if paths is None:
        paths = []

    for path in paths:
        try:
            os.makedirs(path)
        except OSError:
            pass


def train():
    config_path = os.path.join('../', ENCODER, "config_files")

    create_folders([config_path])

    config_path = create_config_file()
    os.system(f'onmt_build_vocab -config {config_path} -n_sample 50000')
    wandb.init(project="ts-mt")
    os.system(f'onmt_train -config {config_path}')


def translate(tgt_corpus):
    pred_path = os.path.join('../' + ENCODER, "prediction")

    model_path = f'../{ENCODER}/run/model_step_{training_steps}.pt'
    test_file = f"{DATASET_DIR}/test/{args.src_corpus}-test.txt"
    create_folders([pred_path])

    if not args.embedding:
        pred_file = os.path.join(pred_path, f"{ENCODER}.{tgt_corpus}-pred.txt")
    else:
        pred_file = os.path.join(pred_path, f"{ENCODER}.{tgt_corpus}-pred.embedding.txt")

    translate_cmd = f'onmt_translate -model {model_path} -src {test_file} -output {pred_file} -verbose '
    if torch.cuda.is_available():
        translate_cmd += ' -gpu 0'
    os.system(translate_cmd)


def evaluate(tgt_corpus):
    pred_path = os.path.join('../' + ENCODER, "prediction")
    if not args.embedding:
        pred_file = os.path.join(pred_path, f"{ENCODER}.{tgt_corpus}-pred.txt")
    else:
        pred_file = os.path.join(pred_path, f"{ENCODER}.{tgt_corpus}-pred.embedding.txt")

    result = {}
    model_dir = os.path.join('..', ENCODER)
    preds = open(os.path.join(model_dir, 'prediction', pred_file), encoding='utf-8').readlines()

    inputs = open(
        os.path.join(DATASET_DIR, 'test', f'{args.src_corpus}-test.txt'),
        encoding='utf-8').readlines()

    result_dict = {
        'src_sent': inputs,
        'pred_sent': preds,
    }

    reference_names = []
    for i, ref_file in enumerate(os.listdir(os.path.join(DATASET_DIR, 'test/references'))):
        if ref_file.split('_')[-1] == tgt_corpus:
            target = open(os.path.join(DATASET_DIR, 'test/references', ref_file), encoding='utf8').readlines()
            reference_names.append(ref_file.split('.')[0])
            result_dict[ref_file.split('.')[0]] = target

    df = pd.DataFrame(result_dict)

    refs = df.loc[:, reference_names].to_numpy()

    def list_bleu(tup):
        return sentence_bleu(sys_sent=tup[0], ref_sents=tup[1])

    list_score = list(map(list_bleu, zip(preds, refs)))

    df['bleu_score'] = list_score
    refs = df.loc[:, reference_names].T.to_numpy()
    bleu_score = corpus_bleu(refs_sents=refs, sys_sents=preds)
    sari_score = corpus_sari(orig_sents=inputs, refs_sents=refs, sys_sents=preds)

    result["result"] = {
        'BLEU': round(bleu_score, 2),
        'SARI': round(sari_score, 2),
    }
    df.to_csv(os.path.join(model_dir, 'reports', f'{args.src_corpus}-{args.tgt_corpus}.sent_report.csv'))
    pd.DataFrame.from_dict(result, orient='index').to_csv(
        os.path.join(model_dir, 'reports', '{args.src_corpus}-{args.tgt_corpus}.corpus_report.csv'))


def main():
    global TARGET_FILES
    for corpus in os.listdir('../datasets/train/'):
        tar = corpus.split('-')[-1]
        # TARGET_FILES = [tar]
        # train()
        translate(tar)
        evaluate(tar)


if __name__ == '__main__':
    main()
