''' Handling the data io '''
import os
import argparse
import logging
import dill as pickle
import urllib
from tqdm import tqdm
import sys
import codecs
import spacy
from torchtext.data import Field, BucketIterator
import tarfile
import torchtext.data
import torchtext.datasets
from torchtext.datasets import TranslationDataset
import numpy as np

__author__ = "Yu-Hsiang Huang"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-raw_dir', required=True)
    parser.add_argument('-data_dir', required=True)
    parser.add_argument('-codes', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-prefix', required=True)
    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('--symbols', '-s', type=int, default=32000, help="Vocabulary size")
    parser.add_argument(
        '--min-frequency', type=int, default=6, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
                        help="If set, input file is interpreted as a dictionary where each line contains a word-count "
                             "pair")
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument('--total-symbols', '-t', action="store_true")
    opt = parser.parse_args()

    # Create folder if needed.
    mkdir_if_needed(opt.raw_dir)
    mkdir_if_needed(opt.data_dir)

    # Download and extract raw data.
    raw_train = get_raw_files(opt.raw_dir, _TRAIN_DATA_SOURCES)
    raw_val = get_raw_files(opt.raw_dir, _VAL_DATA_SOURCES)
    raw_test = get_raw_files(opt.raw_dir, _TEST_DATA_SOURCES)

    # Merge files into one.
    train_src, train_trg = compile_files(opt.raw_dir, raw_train, opt.prefix + '-train')
    val_src, val_trg = compile_files(opt.raw_dir, raw_val, opt.prefix + '-val')
    test_src, test_trg = compile_files(opt.raw_dir, raw_test, opt.prefix + '-test')

    # Build up the code from training files if not exist
    opt.codes = os.path.join(opt.data_dir, opt.codes)
    if not os.path.isfile(opt.codes):
        sys.stderr.write(f"Collect codes from training data and save to {opt.codes}.\n")
        learn_bpe(raw_train['src'] + raw_train['trg'], opt.codes, opt.symbols, opt.min_frequency, True)
    sys.stderr.write(f"BPE codes prepared.\n")

    sys.stderr.write(f"Build up the tokenizer.\n")
    with codecs.open(opt.codes, encoding='utf-8') as codes:
        bpe = BPE(codes, separator=opt.separator)

    sys.stderr.write(f"Encoding ...\n")
    encode_files(bpe, train_src, train_trg, opt.data_dir, opt.prefix + '-train')
    encode_files(bpe, val_src, val_trg, opt.data_dir, opt.prefix + '-val')
    encode_files(bpe, test_src, test_trg, opt.data_dir, opt.prefix + '-test')
    sys.stderr.write(f"Done.\n")

    field = torchtext.data.Field(
        tokenize=str.split,
        lower=True,
        pad_token=Constants.PAD_WORD,
        init_token=Constants.BOS_WORD,
        eos_token=Constants.EOS_WORD)

    fields = (field, field)

    MAX_LEN = opt.max_len

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    enc_train_files_prefix = opt.prefix + '-train'
    train = TranslationDataset(
        fields=fields,
        path=os.path.join(opt.data_dir, enc_train_files_prefix),
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    from itertools import chain
    field.build_vocab(chain(train.src, train.trg), min_freq=2)

    data = {'settings': opt, 'vocab': field, }
    opt.save_data = os.path.join(opt.data_dir, opt.save_data)

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))


def main_wo_bpe():
    """
    Usage: python preprocess.py -lang_src de -lang_trg en -save_data multi30k_de_en.pkl -share_vocab
    """
    glove_path = 'glove_s300.txt'
    spacy_support_langs = ['spt', 'pt']

    parser = argparse.ArgumentParser()
    parser.add_argument('-lang_src', default='pt', choices=spacy_support_langs)
    parser.add_argument('-lang_trg', default='spt', choices=spacy_support_langs)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-data_src', type=str, default=None)
    parser.add_argument('-data_trg', type=str, default=None)

    parser.add_argument('-max_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=3)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-embedding_path', type=str, default=None)
    # parser.add_argument('-ratio', '--train_valid_test_ratio', type=int, nargs=3, metavar=(8,1,1))
    # parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    assert not any([opt.data_src, opt.data_trg]), 'Custom data input is not support now.'
    assert not any([opt.data_src, opt.data_trg]) or all([opt.data_src, opt.data_trg])
    print(opt)

    nlp = spacy.load('pt_core_news_sm')

    def tokenize_pt(text):
        """
        Tokenizes Portuguese text from a string into a list of strings
        """
        return [tok.text for tok in nlp.tokenizer(text)]

    SRC = Field(tokenize = tokenize_pt, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = not opt.keep_case, 
            batch_first = True)

    TRG = Field(tokenize = tokenize_pt, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = not opt.keep_case, 
                batch_first = True)

    MAX_LEN = opt.max_len
    MIN_FREQ = opt.min_word_count

    if not all([opt.data_src, opt.data_trg]):
        assert {opt.lang_src, opt.lang_trg} == {'pt', 'spt'}
    else:
        # Pack custom txt file into example datasets
        raise NotImplementedError

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train_data, valid_data, test_data = torchtext.datasets.TranslationDataset.splits(
        path='ts-corpus-mt/',
        train='train',
        validation='val',
        test='test',
        exts=('.pt', '.spt'),
        fields=(SRC, TRG), filter_pred=filter_examples_with_length)

    SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
    print('[Info] Get source language vocabulary size:', len(SRC.vocab))
    TRG.build_vocab(train_data.trg, min_freq=MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(TRG.vocab))

    if opt.share_vocab:
        print('[Info] Merging two vocabulary ...')
        for w, _ in SRC.vocab.stoi.items():
            # TODO: Also update the `freq`, although it is not likely to be used.
            if w not in TRG.vocab.stoi:
                TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
        TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
        for w, i in TRG.vocab.stoi.items():
            TRG.vocab.itos[i] = w
        SRC.vocab.stoi = TRG.vocab.stoi
        SRC.vocab.itos = TRG.vocab.itos
        print('[Info] Get merged vocabulary size:', len(TRG.vocab))



        if opt.embedding_path:
            word2vec = {}
            print("[Info] Creating Embedding...")
            with open(opt.embedding_path, 'r') as f:
                for l in f:
                    try:
                        line = l.split()
                        word = line[0]
                        vect = np.array(line[1:]).astype(np.float)
                        word2vec[word] = vect
                    except ValueError:
                        pass
            matrix_len = len(SRC.vocab.itos)
            emb_dim = 300
            weights_matrix = np.zeros((matrix_len, emb_dim))
            words_found = 0
            print("[Info] Creating Embedding Matrix...")
            for i, word in enumerate(SRC.vocab.itos):
                try:
                    weights_matrix[i] = word2vec[word]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = np.random.normal(scale=0.6, size=(1, emb_dim))
                except ValueError:
                    weights_matrix[i] = np.random.normal(scale=0.6, size=(1, emb_dim))

            np.save('weights_matrix', weights_matrix)

    data = {
        'settings': opt,
        'vocab': {'src': SRC, 'trg': TRG},
        'train': train_data.examples,
        'valid': valid_data.examples,
        'test': valid_data.examples}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))


if __name__ == '__main__':
    main_wo_bpe()
    # main()
