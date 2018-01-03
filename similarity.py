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

from collections import Counter
import matplotlib.pyplot as plt
import operator

def eda(df):
    print ("Duplicate Count = %s , Non Duplicate Count = %s"
           %(df.is_duplicate.value_counts()[1],df.is_duplicate.value_counts()[0]))

    question_ids_combined = df.qid1.tolist() + df.qid2.tolist()

    print ("Unique Questions = %s" %(len(np.unique(question_ids_combined))))

    question_ids_counter = Counter(question_ids_combined)
    sorted_question_ids_counter = sorted(question_ids_counter.items(), key=operator.itemgetter(1))
    question_appearing_more_than_once = [i for i in question_ids_counter.values() if i > 1]
    print ("Count of Quesitons appearing more than once = %s" %(len(question_appearing_more_than_once)))
eda(train)

def eda(df):
    question_ids_combined = df.qid1.tolist() + df.qid2.tolist()

    print ("Unique Questions = %s" %(len(np.unique(question_ids_combined))))

    question_ids_counter = Counter(question_ids_combined)
    sorted_question_ids_counter = sorted(question_ids_counter.items(), key=operator.itemgetter(1))
    question_appearing_more_than_once = [i for i in question_ids_counter.values() if i > 1]
    print ("Count of Quesitons appearing more than once = %s" %(len(question_appearing_more_than_once)))
eda(test)


import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords

words = re.compile(r"\w+",re.I)
stopword = stopwords.words('english')

def tokenize_questions(df):
    question_1_tokenized = []
    question_2_tokenized = []

    for q in df.question1.tolist():
        question_1_tokenized.append([i.lower() for i in words.findall(q) if i not in stopword])

    for q in df.question2.tolist():
        question_2_tokenized.append([i.lower() for i in words.findall(q) if i not in stopword])

    df["Question_1_tok"] = question_1_tokenized
    df["Question_2_tok"] = question_2_tokenized

    return df

def train_dictionary(df):

    questions_tokenized = df.Question_1_tok.tolist() + df.Question_2_tok.tolist()

    dictionary = corpora.Dictionary(questions_tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000000)
    dictionary.compactify()

    return dictionary

df_train = tokenize_questions(train)
dictionary = train_dictionary(df_train)
print ("No of words in the dictionary = %s" %len(dictionary.token2id))

def get_vectors(df, dictionary):

    question1_vec = [dictionary.doc2bow(text) for text in df.Question_1_tok.tolist()]
    question2_vec = [dictionary.doc2bow(text) for text in df.Question_2_tok.tolist()]

    question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))
    question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))

    return question1_csc.transpose(),question2_csc.transpose()


q1_csc, q2_csc = get_vectors(df_train, dictionary)

df_test = tokenize_questions(test)
dictionary = train_dictionary(df_test)
q1_csc, q2_csc = get_vectors(df_test, dictionary)

from sklearn.metrics.pairwise import cosine_similarity as cs

def get_cosine_similarity(q1_csc, q2_csc):
    cosine_sim = []
    for i,j in zip(q1_csc, q2_csc):
        sim = cs(i,j)
        cosine_sim.append(sim[0][0])

    return cosine_sim

cosine_sim = get_cosine_similarity(q1_csc, q2_csc)

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

np.random.seed(10)

def train_rfc(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    svm_models = [('svm', SVC(verbose=1, shrinking=False))]
    svm_pipeline = Pipeline(svm_models)
    svm_params = {'svm__kernel' : ['rbf'],
                  'svm__C' : [0.01,0.1,1],
                  'svm__gamma' :[0.1,0.2,0.4],
                  'svm__tol' :[0.001,0.01,0.1],
                  'svm__class_weight' : [{1:0.8,0:0.2}]}

    rfc_models = [('rfc', RFC())]
    rfc_pipeline = Pipeline(rfc_models)
    rfc_params = {'rfc__n_estimators' : [40],
                  'rfc__max_depth' : [40],
                  'rfc__min_samples_leaf' : [50]}

    lr_models = [('lr', LR(verbose=1))]
    lr_pipeline = Pipeline(lr_models)
    lr_params = {'lr__C': [0.1, 0.01],
                 'lr__tol': [0.001,0.01],
                 'lr__max_iter': [200,400],
                 'lr__class_weight' : [{1:0.8,0:0.2}]}

    gbc_models = [('gbc', GBC(verbose=1))]
    gbc_pipeline = Pipeline(gbc_models)
    gbc_params = {'gbc__n_estimators' : [100,200, 400, 800],
                  'gbc__max_depth' : [40, 80, 160, 320],
                  'gbc__learning_rate' : [0.01,0.1]}

    grid = zip([svm_pipeline, rfc_pipeline, lr_pipeline, gbc_pipeline],
               [svm_params, rfc_params, lr_params, gbc_params])

    grid = zip([rfc_pipeline],
               [rfc_params])

    best_clf = None

    for model_pipeline, param in grid:
        temp = GridSearchCV(model_pipeline, param_grid=param, cv=4, scoring='f1')
        temp.fit(X_train, y_train)

        if best_clf is None:
            best_clf = temp
        else:
            if temp.best_score_ > best_clf.best_score_:
                best_clf = temp

    model_details = {}
    model_details["CV Accuracy"] = best_clf.best_score_
    model_details["Model Parameters"] = best_clf.best_params_
    model_details["Test Data Score"] = best_clf.score(X_test, y_test)
    model_details["F1 score"] = f1_score(y_test, best_clf.predict(X_test))
    model_details["Confusion Matrix"] = str(confusion_matrix(y_test, best_clf.predict(X_test)))

    return best_clf, model_details
X = np.array(cosine_sim).reshape(-1,1)
y = df_train.is_duplicate

clf, model_details = train_rfc(X,y)

print (model_details)
