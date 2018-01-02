import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import operator
import re
import os
import gc
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import string
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from nltk import word_tokenize, ngrams
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn.metrics import log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
get_ipython().magic('matplotlib inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
pal = sns.color_palette()
color = sns.color_palette()
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', -1)
pd.options.mode.chained_assignment = None  # default='warn'

words = re.compile(r"\w+",re.I)
stopword = stopwords.words('english')


#prelim data exploration
train = pd.read_csv("train.csv").fillna("")
test = pd.read_csv("test.csv").fillna("")
train.groupby("is_duplicate")['id'].count().plot.bar()
dfs = train[0:2500]
dfs.groupby("is_duplicate")['id'].count().plot.bar()

dfq1, dfq2 = dfs[['qid1', 'question1']], dfs[['qid2', 'question2']]
dfq1.columns = ['qid1', 'question']
dfq2.columns = ['qid2', 'question']
dfqa = pd.concat((dfq1, dfq2), axis=0).fillna("")
nrows_for_q1 = dfqa.shape[0]/2

all_ques_df = pd.DataFrame(pd.concat([train['question1'], train['question2']]))
all_ques_df.columns = ["questions"]
all_ques_df["num_of_words"] = all_ques_df["questions"].apply(lambda x : len(str(x).split()))

cnt_srs = all_ques_df['num_of_words'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of words in the question', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

all_ques_df["num_of_chars"] = all_ques_df["questions"].apply(lambda x : len(str(x)))
cnt_srs = all_ques_df['num_of_chars'].value_counts()

plt.figure(figsize=(50,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of characters in the question', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

del all_ques_df

train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)
test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)
plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
plt.title('Normalised histogram of character count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)

print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(),
                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
##########################################
#transform questions with Tf-Tfidf
mq1 = TfidfVectorizer().fit_transform(dfqa['question'].values)
diff_encodings = mq1[::2] - mq1[1::2]

import nltk
STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    regex = re.compile('([^\s\w]|_&*)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence
def clean_trainframe(df):
    df = df.dropna(how="any")

    for col in ['question1', 'question2']:
        df[col] = df[col].apply(clean_sentence)

    return df
def build_corpus(df):
    corpus = []
    for col in ['question1', 'question2']:
        for sentence in df[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)

    return corpus

df = clean_trainframe(train)
corpus = build_corpus(df)

from gensim.models import word2vec
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)

def tsne_plot(model):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model)
