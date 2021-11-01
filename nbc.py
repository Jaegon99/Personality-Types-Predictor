import collections
import csv
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from keras.preprocessing import sequence
from keras.preprocessing import text
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

    model = MultinomialNB()

    k_fold = KFold(n_splits=5)
    scores_k = []
    confusion_k = np.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold.split(x_train):
        x_train_k = df.iloc[train_indices]['x'].values
        y_train_k = df.iloc[train_indices]['y'].values
        x_test_k = df.iloc[test_indices]['x'].values
        y_test_k = df.iloc[test_indices]['y'].values
        model.fit(preprocess(x_train_k), y_train_k)
        predictions_k = model.predict(preprocess(x_test_k))
        predictions_k = np.rint(predictions_k)
        confusion_k += confusion_matrix(y_test_k, predictions_k)
        score_k = accuracy_score(y_test_k, predictions_k)
        scores_k.append(score_k)
    with open('dataset/report/nbc/cross_validation_{}.txt'.format(DICHOTOMY[d]), 'w') as f:
        f.write('*** {}/{} TRAINING SET CROSS VALIDATION (POSTS) ***\n'.format(DICHOTOMY[d][0], DICHOTOMY[d][1]))
        f.write('Total posts classified: {}\n'.format(len(x_train)))
        f.write('Accuracy: {}\n'.format(sum(scores_k) / len(scores_k)))
        f.write('Confusion matrix: \n')
        f.write(np.array2string(confusion_k, separator=', '))
