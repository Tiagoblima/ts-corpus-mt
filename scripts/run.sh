
ENCODER="brnn"

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


python pipeline1.py --encoder $ENCODER  --epochs $N_STEPS

wandb sync ../$ENCODER/runs/fit
