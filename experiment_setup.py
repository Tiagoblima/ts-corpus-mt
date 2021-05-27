import os
from nltk.tokenize import word_tokenize
import torch
import nltk
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)
parser.add_argument('--embedding', action='store_true',
                    help='an integer for the accumulator')

args = parser.parse_args()

nltk.download('punkt')
ENCODER = args.model
os.system('pip3 install openNMT-py')
open('results.txt', 'w')
results_file = open('results.txt', 'a')
results_file.write('dataset,BLEU SCORE\n')

training_steps = 30000


def create_config_file(folder_name_):
    global training_steps
    model_config = open(f'{ENCODER}/{ENCODER}.config.yaml').read()
    if args.embedding:
        emb_config = "both_embeddings: glove_dir/glove_s300.txt\nembeddings_type: \"GloVe\"\nword_vec_size: 300\n\n"

        model_config += emb_config

    source_ = folder_name.split('-')[0]
    target_ = folder_name.split('-')[1]
    config_file_path = os.path.join(ENCODER, 'config_files', f'{ENCODER}.{folder_name_}.yaml')
    file = open(config_file_path, 'w')
    path_to_save = f"save_data: datasets/{folder_name_}/samples\n"
    file.write(path_to_save)
    source_path = f"src_vocab: datasets/{folder_name_}/vocab/portuguese.vocab\n"
    file.write(source_path)
    tgt_path = f"tgt_vocab: datasets/{folder_name_}/vocab/portuguese.vocab\n"
    file.write(tgt_path)
    options = "overwrite: True\nshare_vocab: True\n"
    file.write(options)

    data_str = "data:\n" \
               " corpus_1:\n" \
               f"   path_src: datasets/{folder_name_}/train.{source_}\n" \
               f"   path_tgt: datasets/{folder_name_}/train.{target_}\n" \
               " valid:\n" \
               f"   path_src: datasets/{folder_name_}/val.{source_}\n" \
               f"   path_tgt: datasets/{folder_name_}/val.{target_}\n"
    file.write(data_str)
    model_path = f"save_model: {ENCODER}/run/{folder_name_}/model\n"
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


for folder_name in os.listdir('datasets/'):
    source = folder_name.split('-')[0]
    target = folder_name.split('-')[1]
    try:
        os.makedirs(os.path.join(ENCODER, "reports"))
    except OSError:
        pass

    try:
        os.makedirs(os.path.join(ENCODER, "prediction"))
    except OSError:
        pass

    try:
        os.makedirs(os.path.join(ENCODER, "config_files"))
    except OSError:
        pass

    config_path = create_config_file(folder_name)

    os.system(f'onmt_build_vocab -config {config_path} -n_sample 10000')
    os.system(f'onmt_train -config {config_path}')
    test_file = f"datasets/{folder_name}/test.{source}"
    pred_file = f"{ENCODER}/prediction/{source}-{target}-pred.txt"
    translate_cmd = f'onmt_translate -model {ENCODER}/run/{folder_name}/model_step_{training_steps}.pt -src {test_file} -output {pred_file} -verbose'
    if torch.cuda.is_available():
        translate_cmd += ' -gpu 0'
    os.system(translate_cmd)
    refs = open(f'datasets/{folder_name}/test.{target}', encoding="utf8").readlines()
    refs = list(map(lambda sent: [word_tokenize(sent)], refs))
    inputs = open(f'datasets/{folder_name}/test.{source}', encoding="utf8").readlines()
    hypothesis = open(f'{ENCODER}/prediction/{source}-{target}-pred.txt', encoding='utf8').readlines()

    try:
        BLEUscore = nltk.translate.bleu_score.corpus_bleu(refs, hypothesis)
        results_file.write('{},{:.2f}\n'.format(folder_name, BLEUscore))
        print('{},{:.2f}\n'.format(folder_name, BLEUscore))
    except AssertionError:
        print("Verificar treinamento")

results_file.close()
os.system(f'zip -r {ENCODER}-pred.zip {ENCODER}')
