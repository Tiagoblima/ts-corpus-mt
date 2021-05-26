ssh u63074@login-2
pip install nltk
pip install openNMT-py
pip install numpy==1.20.0
cd ts-corpus-mt/ || exit
python experiment_setup.py --model brnn
#python experiment_setup.py --model transformer
