sh config.sh
cd ts-corpus-mt/ || echo "ts-corpus-mt/ Dir not found"

ENCODER="transformer"
python execute_openmt.py --model transformer
python evaluation.py --model transformer
zip -r transformer-pred.zip ../transformer
wandb sync ../$ENCODER/runs/fit
