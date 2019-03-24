from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from collections import Counter
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import os
import sys
import zipfile
import re

BATCH_SIZE = 64
NUM_EPOCHS = 100
GLOVE_EMBEDDING_SIZE = 100
GLOVE_MODEL = 'glove.6B.100d.txt'
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'
WEIGHT_FILE_PATH = 'word-glove-weights.h5'


def load_glove():                            #loading glove model
    _word2em = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf-8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em


def clean_text(text):                   #cleaning samples
    text = text.lower()
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"i 'm", "i am", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    text = re.sub(r"she 's", "she is", text)
    text = re.sub(r"he 's", "he is", text)
    text = re.sub(r"that 's", "that is", text)
    text = re.sub(r"what 's", "what is", text)
    text = re.sub(r"where 's", "where is", text)
    text = re.sub(r"don't", 'do not', text)
    text = re.sub(r"n't", 'not', text)
    text = re.sub(r"gon na", 'going to', text)
    text = re.sub(r"gonna", 'going to', text)
    text = re.sub(r"wanna", 'want to', text)
    text = re.sub(r"dunno", 'do not know', text)
    text = re.sub(r"gotta", 'got to', text)
    text = re.sub(r"gonna", 'going to', text)
    text = re.sub(r"\*", '', text)
    text = re.sub(r'/u/', '', text)
    text = re.sub(r'\[]', '', text)
    text = re.sub(r'%', '', text)
    text = re.sub(r'~', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'$', '', text)
    text = re.sub(r'@', '', text)
    text = re.sub(r'&', '', text)
    text = re.sub(r'^', '', text)
    text = re.sub(r'newlinechar', '', text)
    return text


def generate_batch(input_word2em_data, output_text_data, target_word2idx, word2em, decoder_max_seq_length, encoder_max_seq_length, num_decoder_tokens): 
    num_batches = len(input_word2em_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_word2em_data[start:end], encoder_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, GLOVE_EMBEDDING_SIZE))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = target_word2idx['unknown']  # default unknown
                    if w in target_word2idx:
                        w2idx = target_word2idx[w]
                    if w in word2em:
                        decoder_input_data_batch[lineIdx, idx, :] = word2em[w]
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch
