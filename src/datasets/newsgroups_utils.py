from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from collections import Counter


import numpy as np
import re

newsgroup20_categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']


lexicon = (
    (re.compile(r"\bdon't\b"), "do not"),
    (re.compile(r"\bit's\b"), "it is"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"\bi've\b"), "i have"),
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bdoesn't\b"), "does not"),
    (re.compile(r"\bthat's\b"), "that is"),
    (re.compile(r"\bdidn't\b"), "did not"),
    (re.compile(r"\bi'd\b"), "i would"),
    (re.compile(r"\byou're\b"), "you are"),
    (re.compile(r"\bisn't\b"), "is not"),
    (re.compile(r"\bi'll\b"), "i will"),
    (re.compile(r"\bthere's\b"), "there is"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bwoudn't\b"), "would not"),
    (re.compile(r"\bhe's\b"), "he is"),
    (re.compile(r"\bthey're\b"), "they are"),
    (re.compile(r"\bwe're\b"), "we are"),
    (re.compile(r"\blet's\b"), "let us"),
    (re.compile(r"\bhaven't\b"), "have not"),
    (re.compile(r"\bwhat's\b"), "what is"),
    (re.compile(r"\baren't\b"), "are not"),
    (re.compile(r"\bwasn't\b"), "was not"),
    (re.compile(r"\bwouldn't\b"), "would not"),
)

def fix_apostrophes(text):
    text = text.lower()
    
    for pattern, replacement in lexicon:
        text = pattern.sub(replacement, text)

    return text

def get_newsgroups(): 
    
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    newsgroups_data = fetch_20newsgroups(subset='all', shuffle=False, 
                                      categories=newsgroup20_categories,)

    labels=newsgroups_data.target
    texts = newsgroups_data.data

    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000

    texts = list(map(fix_apostrophes, texts))
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
        lower=False, 
        filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')

    return data, labels, word_index


def glove_embeddings(glove_path, word_index):
    embeddings_index = {}

    with open(glove_path) as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            #values[-1] = values[-1].replace('\n', '')
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            #print (values[1:])
    
    EMBEDDING_DIM = 100

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        #embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix