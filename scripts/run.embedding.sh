
ENCODER="brnn"

N_STEPS=10000
cd indigenous-mt/scripts/ || echo "scripts/ Dir not found"
pip install -r ../requirements.txt
pip install wandb -qqq

git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py && pip install -e .
cd ../

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

python pipeline.py --encoder $ENCODER  --epochs $N_STEPS --embedding

wandb sync ../$ENCODER/runs/fit
