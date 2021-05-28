cd ts-corpus-mt/ || exit
python experiment_setup.py --model brnn
python evaluation.py --model brnn
zip -r brnn-pred.zip brnn
zip -r brnn.embedding-pred.zip brnn
