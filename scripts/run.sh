
ENCODER="brnn"
SOURCE_LANG="complex"
TARGET_LANG="simple"
N_STEPS=60000
cd indigenous-mt/scripts/ || echo "scripts/ Dir not found"
pip install -r ../requirements.txt
pip install wandb -qqq

if [ -d "../datasets/"$SOURCE_LANG-$TARGET_LANG ]
then
     echo "../datasets dir exists"
else
    echo "CREATING DATAFILES"
    python preprocess.py --src_lang $SOURCE_LANG --tgt_lang $TARGET_LANG
fi


python train.py --encoder $ENCODER  --epochs $N_STEPS --src_lang $SOURCE_LANG --tgt_lang $TARGET_LANG
%tensorboard --logdir ../$ENCODER/runs/fit
python translate.py --encoder $ENCODER --model model_step_$N_STEPS.pt
python evaluation.py --encoder $ENCODER --src_lang $SOURCE_LANG --tgt_lang $TARGET_LANG

wandb sync ../$ENCODER/runs/fit
