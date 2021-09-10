import tweepy as tw
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing import sequence
from keras.models import load_model

MAX_POST_LENGTH = 40

DICHOTOMY = ('IE', 'NS', 'TF', 'PJ')
types = ['infj', 'entp', 'intp', 'intj',
         'entj', 'enfj', 'infp', 'enfp',
         'isfp', 'istp', 'isfj', 'istj',
         'estp', 'esfp', 'estj', 'esfj']

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")


def lemmatize(x):
    lemmatized = []
    for post in x:
        temp = post.lower()
        for type_ in types:
            temp = temp.replace(' ' + type_, '')
        temp = ' '.join([lemmatizer.lemmatize(word) for word in temp.split(' ') if (word not in stop_words)])
        lemmatized.append(temp)
    return np.array(lemmatized)


consumer_key = '8YCOrCne0omjoGKmFYts0zsRx'
consumer_secret = 'BOHdap8jS6wbHI4oaHyn0XPsVqKXJcbQFXeNwcCJJYhWshE2OK'
access_token = '1430845309924610050-PPsrLh5BVqzxWrdJbYse7erMAWUeGw'
access_token_secret = 'yEAFTN50WNTZxAAdF3QPDo5pz4KGajCZbl5XKgV8eE9Bn'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


def getType(handle):
    res = api.user_timeline(screen_name=handle, count=100, include_rts=False)
    tweets = [tweet.text for tweet in res]

    x_test = []
    for tweet in tweets:
        x_test.append(tweet)

    yourType = ''
    for d in range(len(DICHOTOMY)):
        model = load_model('dataset/model/model_{}.h5'.format(DICHOTOMY[d]))
        tokenizer = None
        with open('dataset/model/tokenizer_{}.pkl'.format(DICHOTOMY[d]), 'rb') as f:
            tokenizer = pickle.load(f)

        def preprocess(x):
            lemmatized = lemmatize(x)
            tokenized = tokenizer.texts_to_sequences(lemmatized)
            return sequence.pad_sequences(tokenized, maxlen=MAX_POST_LENGTH)

        predictions = model.predict(preprocess(x_test))
        prediction = float(sum(predictions) / len(predictions))
        if prediction >= 0.5:
            yourType += DICHOTOMY[d][1]
        else:
            yourType += DICHOTOMY[d][0]

    return yourType
