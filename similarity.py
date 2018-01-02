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
