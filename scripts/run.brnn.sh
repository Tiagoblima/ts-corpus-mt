sh config.sh
cd ts-corpus-mt/ || echo "ts-corpus-mt/ Dir not found"
python execute_experiment.py --model brnn
python evaluation.py --model brnn
zip -r brnn-pred.zip ../brnn
