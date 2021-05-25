import os
from nltk.tokenize import word_tokenize
import torch
import nltk

nltk.download('punkt')
ENCODER = "transformer"
os.system('pip3 install openNMT-py')
# open('results.txt', 'w')
results_file = open('results.txt', 'a')
results_file.write('dataset,BLEU SCORE')

training_steps = 5000


def create_config_file(folder_name_):
    model_config = open(f'{ENCODER}.config.yaml').read()
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
    options = "overwrite: False\nshare_vocab: True\n"
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
        training_steps = 5000
        file.write(f"\nsave_checkpoint_steps: {training_steps}\ntrain_steps: {training_steps}")
        file.write('\ngpu_ranks: [0]\n')
        file.write("batch_size: 16\nvalid_batch_size: 16")
    else:
        training_steps = 1000
        file.write(f"\nsave_checkpoint_steps: {training_steps}\ntrain_steps: {training_steps}")
        file.write("\nbatch_size: 64\nvalid_batch_size: 64")
    file.close()
    return config_file_path


for folder_name in os.listdir('datasets/'):
    source = folder_name.split('-')[0]
    target = folder_name.split('-')[1]
    try:
        os.makedirs(os.path.join(ENCODER, "config_files"))
        os.makedirs(os.path.join(ENCODER, "prediction"))
        os.makedirs(os.path.join(ENCODER, "reports"))

    except OSError:
        pass

    config_path = create_config_file(folder_name)

    os.system(f'onmt_build_vocab -config {config_path} -n_sample 10000')
    os.system(f'onmt_train -config {config_path}')
    test_file = f"datasets/{folder_name}/test.{source}"
    pred_file = f"{ENCODER}/prediction/{source}-{target}-pred.txt"
    os.system(
        f'onmt_translate -model {ENCODER}/run/{folder_name}/model_step_{training_steps}.pt -src {test_file} -output {pred_file} -gpu 0 -verbose')

    refs = open(f'datasets/{folder_name}/test.{target}').readlines()
    refs = list(map(lambda sent: [word_tokenize(sent)], refs))
    inputs = open(f'datasets/{folder_name}/test.{source}').readlines()
    hypothesis = open(f'{ENCODER}/prediction/{source}-{target}-pred.txt').readlines()
    hypothesis = list(map(lambda hyp: word_tokenize(hyp), hypothesis))
    BLEUscore = nltk.translate.bleu_score.corpus_bleu([refs], hypothesis)
    results_file.write('{},{:.2f}\n'.format(folder_name, BLEUscore))

results_file.close()
os.system(f'zip -r {ENCODER}-pred.zip {ENCODER}')
