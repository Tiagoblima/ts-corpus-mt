
ENCODER="brnn"
TEST_DIR=../datasets/test/references
N_STEPS=30000
cd indigenous-mt/scripts/ || echo "scripts/ Dir not found"
pip install -r ../requirements.txt
pip install wandb -qqq

git clone https://github.com/feralvam/easse.git
cd easse || echo "no easse dir found"
pip install .
cd ../


if [ -d "../datasets/" ]
then
     echo "../datasets dir exists"
else
    echo "CREATING DATAFILES"
    python preprocess.py
fi

if [ -d "../glove_dir" ]
then
     echo "Glove dir exists"
else
    mkdir "../glove_dir"
    wget -O glove_s300.zip  https://www.dropbox.com/s/s74ovzynh5jbccz/glove_s300.zip?dl=1
    unzip glove_s300.zip -d "../glove_dir"
fi

python train.py --encoder $ENCODER  --epochs $N_STEPS --embedding

python translate.py --encoder $ENCODER --model model_step_$N_STEPS.pt --embedding
easse evaluate -t custom --orig_sents_path ../datasets/test/src-test.txt --refs_sents_paths $TEST_DIR/reference_1.txt,$TEST_DIR/reference_2.txt,$TEST_DIR/reference_3.txt,$TEST_DIR/reference_4.txt -m 'bleu,sari' -q < ../$ENCODER/prediction/prediction.txt > ../$ENCODER/$ENCODER.reports.txt
wandb sync ../$ENCODER/runs/fit