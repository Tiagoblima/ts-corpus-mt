import argparse
import os

import nltk
import torch
import wandb

wandb.login(key="8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4")

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--encoder', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

parser.add_argument('--epochs', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

parser.add_argument('--embedding', action='store_true',
                    help='an integer for the accumulator')

parser.add_argument('--src_lang', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)
parser.add_argument('--tgt_lang', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

args = parser.parse_args()

nltk.download('punkt')
ENCODER = args.encoder
ROOT_DIR = f'../{ENCODER}'
training_steps = args.epochs

DATASET_DIR = '../datasets/'

TARGET_LANG = args.tgt_lang.lower()
SOURCE_LANG = args.src_lang.lower()


def create_config_file(folder_name_):
    global training_steps

    model_config = open(f'../{ENCODER}/{ENCODER}.config.yaml').read()
    data_config = open(os.path.join(DATASET_DIR, folder_name_, 'data.config.yaml')).read()
    if args.embedding:
        emb_config = "both_embeddings: ../glove_dir/glove_s300.txt\nembeddings_type: \"GloVe\"\nword_vec_size: 300\n\n"

        model_config += emb_config

    config_file_path = os.path.join('../', ENCODER, 'config_files', f'{ENCODER}.{folder_name_}.yaml')

    file = open(config_file_path, 'w')
    logs_path = os.path.join(ROOT_DIR, 'runs/fit')
    file.write(f"tensorboard_log_dir: {logs_path}\n")
    wandb.tensorboard.patch(root_logdir=logs_path)
    model_path = f"save_model: ../{ENCODER}/run/{folder_name_}/model\n"
    file.write(model_path)
    file.write(data_config)
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


def main():
    config_path = os.path.join('../', ENCODER, "config_files")

    create_folders([config_path])

    folder_name = '-'.join([SOURCE_LANG, TARGET_LANG])
    config_path = create_config_file(folder_name)

    os.system(f'onmt_build_vocab -config {config_path} -n_sample 10000')
    wandb.init(project="indigenous-mt")
    os.system(f'onmt_train -config {config_path}')


if __name__ == '__main__':
    main()
