import collections
import csv
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

DICHOTOMY = ('IE', 'NS', 'TF', 'PJ')
TYPES = ['infj', 'entp', 'intp', 'intj',
         'entj', 'enfj', 'infp', 'enfp',
         'isfp', 'istp', 'isfj', 'istj',
         'estp', 'esfp', 'estj', 'esfj']

MODEL_BATCH_SIZE = 128
TOP_WORDS = 2500
MAX_POST_LENGTH = 40
EMBEDDING_VECTOR_LENGTH = 50
LEARNING_RATE = 0.01
DROPOUT = 0.1
NUM_EPOCHS = 30

for d in range(len(DICHOTOMY)):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    with open('dataset/train-set/train{}.csv'.format(DICHOTOMY[d][0]), 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for post in row:
                x_train.append(post)
                y_train.append(0)

    with open('dataset/train-set/train{}.csv'.format(DICHOTOMY[d][1]), 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for post in row:
                x_train.append(post)
                y_train.append(1)

    with open('dataset/test-set/test{}.csv'.format(DICHOTOMY[d][0]), 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for post in row:
                x_test.append(post)
                y_test.append(0)

    with open('dataset/test-set/test{}.csv'.format(DICHOTOMY[d][1]), 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for post in row:
                x_test.append(post)
                y_test.append(1)

        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words("english")


    def lemmatize(x):
        lemmatized = []
        for post in x:
            temp = post.lower()
            for type in TYPES:
                temp = temp.replace(' ' + type, '')
            temp = ' '.join([lemmatizer.lemmatize(word) for word in temp.split(' ') if (word not in stop_words)])
            lemmatized.append(temp)
        return np.array(lemmatized)


    tokenizer = text.Tokenizer(num_words=TOP_WORDS, split=' ')
    tokenizer.fit_on_texts(lemmatize(x_train))


    def preprocess(x):
        lemmatized = lemmatize(x)
        tokenized = tokenizer.texts_to_sequences(lemmatized)
        return sequence.pad_sequences(tokenized, maxlen=MAX_POST_LENGTH)


    df = pd.DataFrame(data={'x': x_train, 'y': y_train})
    df = df.sample(frac=1).reset_index(drop=True)

    embeddings_index = dict()
    with open('dataset/glove.6B.50d.txt', 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')

    embedding_matrix = np.zeros((TOP_WORDS, EMBEDDING_VECTOR_LENGTH))
    for word, i in tokenizer.word_index.items():
        if i < TOP_WORDS:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(TOP_WORDS, EMBEDDING_VECTOR_LENGTH, input_length=MAX_POST_LENGTH, weights=[embedding_matrix], mask_zero=True, trainable=True))
    model.add(LSTM(EMBEDDING_VECTOR_LENGTH, dropout=DROPOUT, recurrent_dropout=DROPOUT, activation='sigmoid', kernel_initializer='zeros'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(preprocess(df['x'].values), df['y'].values, epochs=NUM_EPOCHS, batch_size=MODEL_BATCH_SIZE)
    predictions = model.predict(preprocess(x_test))
    predictions = np.rint(predictions)
    confusion = confusion_matrix(y_test, predictions)
    score = accuracy_score(y_test, predictions)
    with open('dataset/report/test_set_post_classification_{}.txt'.format(DICHOTOMY[d]), 'w', encoding="utf-8") as f:
        f.write('*** {}/{} TEST SET CLASSIFICATION (POSTS) ***\n'.format(DICHOTOMY[d][0], DICHOTOMY[d][1]))
        f.write('Total posts classified: {}\n'.format(len(x_test)))
        f.write('Accuracy: {}\n'.format(score))
        f.write('Confusion matrix: \n')
        f.write(np.array2string(confusion, separator=', '))

    model.save('dataset/model/model_{}.h5'.format(DICHOTOMY[d]))
    with open('dataset/model/tokenizer_{}.pkl'.format(DICHOTOMY[d]), 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
