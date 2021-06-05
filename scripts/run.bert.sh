sh config.sh
cd ts-corpus-mt/ || echo "ts-corpus-mt/ Dir not found"
python bert.py
python evaluation.py --model bert
zip -r bert-pred.zip ../bert
