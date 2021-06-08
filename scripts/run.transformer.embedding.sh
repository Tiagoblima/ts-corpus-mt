sh config.sh
cd ts-corpus-mt/ || echo "ts-corpus-mt/ Dir not found"
ENCODER="transformer"
if [ -d "../glove_dir" ]
then
     echo "Glove dir exists"
else
    mkdir "../glove_dir"
    wget -O glove_s300.zip http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip
    unzip glove_s300.zip -d "../glove_dir"
fi

python -O execute_openmt.py --model transformer --embedding
python evaluation.py --model transformer.embedding
zip -r transformer.embedding-pred.zip ../transformer.embedding
wandb sync ../$ENCODER/runs/fit
