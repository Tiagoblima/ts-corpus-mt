import argparse
import os

import torch

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--encoder', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)
parser.add_argument('--model', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)
parser.add_argument('--embedding', action='store_true',
                    help='an integer for the accumulator')

parser.add_argument('--src_lang', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)
parser.add_argument('--tgt_lang', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)

args = parser.parse_args()
ENCODER = args.encoder
model_name = args.model
DATASET_ROOT = '../datasets'

TARGET_LANG = args.tgt_lang.lower()
SOURCE_LANG = args.src_lang.lower()



def create_folders(paths=None):
    if paths is None:
        paths = []

    for path in paths:
        try:
            os.makedirs(path)
        except OSError:
            pass


folder_name = '-'.join([SOURCE_LANG, TARGET_LANG])
pred_path = os.path.join('../' + ENCODER, "prediction")

model_path = f'../{ENCODER}/run/{folder_name}/{model_name}'
test_file = f"{DATASET_ROOT}/{folder_name}/test/{SOURCE_LANG}.txt"
create_folders([pred_path])
pred_file = os.path.join(pred_path, f"{SOURCE_LANG}-{TARGET_LANG}-pred.txt")

translate_cmd = f'onmt_translate -model {model_path} -src {test_file} -output {pred_file} -verbose '
if torch.cuda.is_available():
    translate_cmd += ' -gpu 0'

os.system(translate_cmd)
