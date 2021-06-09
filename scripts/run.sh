
ENCODER="brnn"
TEST_DIR=../datasets/test/references
N_STEPS=10000
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


python train.py --encoder $ENCODER  --epochs $N_STEPS
%tensorboard --logdir ../$ENCODER/runs/fit
python translate.py --encoder $ENCODER --model model_step_$N_STEPS.pt
easse evaluate -t custom --orig_sents_path ../datasets/test/src-test.txt --refs_sents_paths $TEST_DIR/reference_1.txt,$TEST_DIR/reference_2.txt,$TEST_DIR/reference_3.txt,$TEST_DIR/reference_4.txt -m 'bleu,sari' -q < ../$ENCODER/prediction/prediction.txt > ../$ENCODER/reports.txt
wandb sync ../$ENCODER/runs/fit
