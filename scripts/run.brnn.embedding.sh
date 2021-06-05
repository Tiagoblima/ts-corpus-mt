sh config.sh
cd ts-corpus-mt/ || exit
if [ -d "../glove_dir" ]
then
     echo "Glove dir exists"
else
    mkdir "../glove_dir"
    wget -O glove_s300.zip http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip
    unzip glove_s300.zip -d "../glove_dir"
fi

python -O execute_openmt.py --model brnn --embedding
python evaluation.py --model brnn
zip -r brnn.embedding-pred.zip ../brnn.embedding
#python execute_experiment.py --model transformer
