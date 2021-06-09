
ENCODER="brnn"

N_STEPS=10000
cd indigenous-mt/scripts/ || echo "scripts/ Dir not found"
pip install -r ../requirements.txt
pip install wandb -qqq

git clone https://github.com/feralvam/easse.git
cd easse || echo "no easse dir found"
pip install .


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
python evaluation.py --encoder $ENCODER

wandb sync ../$ENCODER/runs/fit
