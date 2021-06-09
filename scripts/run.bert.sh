ENCODER="bert"
TEST_DIR=../datasets/test/references
pip install simpletransformers
python bert.py
easse evaluate -t custom --orig_sents_path ../datasets/test/src-test.txt --refs_sents_paths $TEST_DIR/reference_1.txt,$TEST_DIR/reference_2.txt,$TEST_DIR/reference_3.txt,$TEST_DIR/reference_4.txt -m 'bleu,sari' -q < ../$ENCODER/prediction/prediction.txt > ../$ENCODER/$ENCODER.reports.txt
