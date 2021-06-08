sh config.sh

ENCODER="brnn"

cd ts-corpus-mt/ || echo "ts-corpus-mt/ Dir not found"
python execute_openmt.py --model brnn
python evaluation.py --model brnn
zip -r brnn-pred.zip ../brnn
wandb sync ../$ENCODER/runs/fit
