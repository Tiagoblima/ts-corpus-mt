sh config.sh
cd ts-corpus-mt/ || echo "ts-corpus-mt/ Dir not found"
python execute_experiment.py --model transformer
python evaluation.py --model transformer
zip -r transformer-pred.zip transformer
