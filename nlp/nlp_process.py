#-*-coding: utf-8 -*-
__author__ = 'nishimuuu'
import gensim, MeCab, collections
import csv
import sys
from itertools import chain
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_validation import  train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
import os
import re
from gensim.models import word2vec

# write code here...


class NLPExecutor:

  #parameter to filter bow
  no_below = 10
  no_above = 0.3
  sys_path = os.path.dirname(os.path.abspath(__file__))

  def __init__(self):

    self.patSep = re.compile(sep)

  def setModel(self,path):
    self.modelpath = path


  def set_tokens(self,tokens, key=None , path = None):
    if key is None:
      key = xrange(0, len(tokens)-1)
    elif len(key) != len(tokens):
      raise 'Invalid Length Erro'
    self.dict_to_morph = {k:v for (k,v) in zip(key, tokens)}
    self.tokens = tokens
    self.prepare_dictionary([self.dict_to_morph[key] for key in self.dict_to_morph], path)

  def prepare_dictionary(self, tokens, path = None):
    self.dictionary = gensim.corpora.Dictionary(tokens)
    if not path is None:
      self.dictionary.save_as_text(path)

  def separate(self, path,remove_stopwords=False):

    # key: documents => key: [tokens]
    self.dict_to_morph = {key:self.run_mecab(self.documents[key], remove_stopwords) for key in self.documents}

    self.prepare_dictionary([self.dict_to_morph[key] for key in self.dict_to_morph], path)




  def makeBow(self):


    # key: tokens => [[(bows)]]
    self.bow = [self.dictionary.doc2bow(tokens) for tokens in [self.dict_to_morph[key] for key in self.dict_to_morph]]



  def aggregate_words(self,list):
    return dict(collections.Counter(list))

  def lda(self, path, num_topics = 200, iter=10000):
    lda = gensim.models.LdaModel(corpus=self.bow, id2word=self.dictionary,num_topics=num_topics,iterations=iter)
    lda.save(path)
    return lda

  def word2vec(self,path):
    sentences = [self.dict_to_morph[token] for token in self.dict_to_morph]
    model = gensim.models.Word2Vec(sentences= sentences,size=200,window=5,min_count=2)
    model.save(path)
    return model

  def get_tokens_dict(self):
    return self.dict_to_morph

  def predict_topic(self, lda_model, token, dictionary=None, n_topic_words = 5):
    np_topic_features = self.convert_topic_vec(lda_model,token)

    idx = np.argsort(np_topic_features[:,1])
    idx = idx[::-1]
    np_topic_features = np_topic_features[idx]

    topic_str_list = lda_model.print_topic(np_topic_features[0,0],n_topic_words)
    if n_topic_words == 1:
        return [topic_str_list.split('*')[1]]
    else:
        return [t.split('*')[1] for t in topic_str_list.split('+')]

  def convert_topic_vec(self, lda_model, token, dictionary=None):
    #if token given list
    if dictionary is None:
      dictionary = self.dictionary

    token = dictionary.doc2bow(token)

    #token given dictionary all patterns
    topic_array = np.concatenate(
      (np.arange(lda_model.num_topics),np.zeros(lda_model.num_topics)))\
      .reshape(2,lda_model.num_topics)\
      .T
    topic_list = np.array([list(topic) for topic in lda_model[token]])
    for i, topic in enumerate(topic_list):
      topic_array[i][1] = topic[1]

    return topic_array


  def set_dictionary(self, path):
    if isinstance(path, str):
      self.dictionary = gensim.corpora.Dictionary.load_from_text(path)
    else:
      self.dictionary = path


if __name__ == '__main__':
  path = ''
  y_path = ''

  lda_dict_path  = './lda.dict'
  lda_model_path = './lda.model'
  w2v_model_path = './w2v.model'

  separater = Separator()
  separater.read(path)
  separater.separate(lda_dict_path)
  separater.makeBow()
  separater.lda(lda_model_path)
  separater.word2vec(w2v_model_path)

  # regr = linear_model.LogisticRegression(penalty='l1')
  regr = linear_model.Lasso()
  separater.regression(y_path,'./lr.model',regr)








