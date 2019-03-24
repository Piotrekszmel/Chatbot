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

from chatbot_helper import load_glove, clean_text, generate_batch

np.random.seed(42)

BATCH_SIZE = 512               #setting parameters
NUM_EPOCHS = 2
GLOVE_EMBEDDING_SIZE = 100
HIDDEN_UNITS = 256
MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
MAX_VOCAB_SIZE = 10000
DATA_SET_NAME = 'cornell'
DATA_PATH = 'train_data.txt'

GLOVE_MODEL = 'glove.6B.100d.txt'
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'
WEIGHT_FILE_PATH = 'word-glove-weights.h5'


word2em = load_glove()
target_counter = Counter()

input_texts = []
target_texts = []

lines = open(DATA_PATH, 'rt', encoding='utf-8').read().split('\n')

prev_words = []

for i, line in enumerate(lines):
    if i == 1000000:                    #Using 1000000 samples
        break
    
    next_words = [w.lower() for w in nltk.word_tokenize(line)]
    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
        next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]
        
    if len(prev_words) > 0:                    #getting "start" at the beggining and "end" at the end of sentence
        input_texts.append(prev_words)
        target_words = next_words[:]
        target_words.insert(0, 'start')
        target_words.append('end')
        for w in target_words:
            target_counter[w] += 1
        target_texts.append(target_words)
        
    prev_words = next_words
 
clean_input_texts = []
clean_target_texts = []                   
for line in input_texts:                #cleaning all sentences
    line = ' '.join(line)
    line = clean_text(line)
    line = [w.lower() for w in nltk.word_tokenize(line)]
    clean_input_texts.append(line)
for line in target_texts:
    line = ' '.join(line)
    line = clean_text(line)
    line = [w.lower() for w in nltk.word_tokenize(line)]
    clean_target_texts.append(line)


    target_word2idx = dict()
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):             
    target_word2idx[word[0]] = idx + 1                          #creating word to index dict
    
    if 'unknown' not in target_word2idx:
        target_word2idx['unknown'] = 0
    target_idx2word = dict([idx, word] for word, idx in target_word2idx.items())  #creating idx to word dict
    
    num_decoder_tokens = len(target_idx2word) + 1

np.save('word-glove-target-word2idx.npy', target_word2idx)
np.save('word-glove-target-idx2word.npy', target_idx2word)

input_texts_word2em = []

encoder_max_seq_length = 0
decoder_max_seq_length = 0

for input_words, target_words in zip(clean_input_texts, clean_target_texts):
    encoder_input_wids = []
    for w in input_words:
        emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)  
        if w in word2em:
            emb = word2em[w]
        encoder_input_wids.append(emb)                              
    input_texts_word2em.append(encoder_input_wids)                      
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)
    
context = dict()
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

#print(context)
np.save('word-glove-context.npy', context) 


encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')               #Seq2seq model architecture
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


Xtrain, Xtest, Ytrain, Ytest = train_test_split(input_texts_word2em, clean_target_texts, test_size=0.2, random_state=42)

train_gen = generate_batch(Xtrain, Ytrain, word2em=word2em, target_word2idx=target_word2idx, decoder_max_seq_length=decoder_max_seq_length, encoder_max_seq_length=encoder_max_seq_length, num_decoder_tokens=num_decoder_tokens)
test_gen = generate_batch(Xtest, Ytest, word2em=word2em, target_word2idx=target_word2idx, decoder_max_seq_length=decoder_max_seq_length, encoder_max_seq_length=encoder_max_seq_length, num_decoder_tokens=num_decoder_tokens)

train_num_batches = len(Xtrain) // BATCH_SIZE
test_num_batches = len(Xtest) // BATCH_SIZE

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])

model.save_weights(WEIGHT_FILE_PATH)    