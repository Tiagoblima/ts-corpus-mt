import argparse
import os
import json

# TEST_DIR=../datasets/references/references
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--encoder', metavar='N', type=str,
                    help='an integer for the accumulator', required=True)
args = parser.parse_args()
ENCODER = args.encoder
TEST_DIR = "../datasets/test"
ORIGINA_SENT_PATH = f"{TEST_DIR}/src-references.txt"
evaluate_cmd = f"easse evaluate -t custom --orig_sents_path {ORIGINA_SENT_PATH} "
evaluate_cmd += f" --refs_sents_paths {TEST_DIR}/references/reference_1.txt,{TEST_DIR}/references/reference_2.txt," \
                f"{TEST_DIR}/references/reference_3.txt,{TEST_DIR}/references/reference_4.txt "
evaluate_cmd += f"-m bleu,sari -q < ../{ENCODER}/prediction/prediction.txt"
result = os.popen(evaluate_cmd).read()
print(result)
result = os.popen(evaluate_cmd).read()
print(os.system(evaluate_cmd))
json.dump(json.loads(result), open("result.json", "w", encoding="utf8"), indent=4)
