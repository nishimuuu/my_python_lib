#-*-coding: utf-8 -*-
__author__ = '01010357'
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


class Executor:

  #parameter to filter bow
  no_below = 10
  no_above = 0.3
  sys_path = os.path.dirname(os.path.abspath(__file__))

  stop_words_list = ['これ','それ','あれ','この','その','あの','ここ','そこ','あそこ','こちら','どこ','だれ','なに','なん',
                '何','です','あります','おります',
                'います','は','が','の','に','を','で','え','から',
                'まで','より','も','どの','と','し','それで','しかし']
  pos_list = ['動詞','名詞','形容詞','副詞','感動詞']
  def __init__(self):
    self.patUrl = re.compile("https?://[\w/:%#\$&\?\(\)~\.=\+\-]+")
    self.patXml = re.compile("<(\".*?\"|\'.*?\'|[^\'\"])*?>")
    sep = '[!-/:-@[-`{-~、◞⤴○▿゚д◟。♡٩ωو°！？（）〈〉【】『』／≦＜＼≧＞≪≫《》∀〔〕━──\n¥〜∵∴́ ❤⇒→⇔\│←↑↓┃★☆「」・♪～〓◆◇■□▽△▲●〇▼◎．”“※♥́́́]'

    self.patSep = re.compile(sep)

  def setModel(self,path):
    self.modelpath = path

  def read(self,path):
    csvFile = csv.reader(open(path))
    documents = {}

    print 'START:: read document'
    for row in csvFile:
      item = [item.decode('utf-8') for item in row]
      documents[item[0]] = u'\n'.join(item[4:10]).replace(u'。',u'\n')
    self.documents = documents
    print 'FINISH:: read document'

  def run_mecab(self,text, remove_stopwords=False):
    mb = MeCab.Tagger('mecabrc')
    target_text = text.encode('utf-8')
    morphs = mb.parseToNode(target_text)
    resultTokens = []
    while(morphs):
      text_before_encode = morphs.surface
      features = morphs.feature.split(',')
      pos = features[0]


      if pos == 'BOS/EOS' or pos not in Separator.pos_list or text_before_encode in Separator.stop_words_list:
        morphs = morphs.next
        continue

      decoded_text = text_before_encode.decode('utf-8')
      resultTokens.append(decoded_text)
      morphs = morphs.next

    return resultTokens

  def prepare_dictionary(self, tokens,path = None):
    self.dictionary = gensim.corpora.Dictionary(tokens)
    if not path is None:
      self.dictionary.save_as_text(path)

  def separate(self, path,remove_stopwords=False):

    # key: documents => key: [tokens]
    print 'START:: run mecab'
    self.dict_to_morph = {key:self.run_mecab(self.documents[key], remove_stopwords) for key in self.documents}
    print 'FINISH:: run mecab for all documents'
    print 'START:: make dictionary for all documents'

    self.prepare_dictionary([self.dict_to_morph[key] for key in self.dict_to_morph], path)


    print 'FINISH:: make dictionary for all documents'


  def makeBow(self):
    print 'START:: make bow'


    # key: tokens => [[(bows)]]
    self.bow = [self.dictionary.doc2bow(tokens) for tokens in [self.dict_to_morph[key] for key in self.dict_to_morph]]

    print 'FINISH:: make bow'


  def aggregate_words(self,list):
    return dict(collections.Counter(list))

  def lda(self, path):
    print 'START:: lda'
    lda = gensim.models.LdaModel(corpus=self.bow, id2word=self.dictionary,num_topics=200)
    print 'FINISH:: lda'
    lda.save(path)
    self.lda = lda
   
  def word2vec(self,path):
    print 'START:: word2vec'
    # self.token_list = chain.from_iterable([self.dict_to_morph[token] for token in self.dict_to_morph]))
    sentences = [self.dict_to_morph[token] for token in self.dict_to_morph]
    model = gensim.models.Word2Vec(sentences= sentences,size=200,window=5,min_count=2)
    model.save(path)
    print 'FINISH:: word2vec'
    self.w2v = model

  def regression(self, target_path,save_path,regr):
    print 'START:: LogisticRegression'
    listing_dict = {}
    reader = csv.reader(open(target_path,'rb'))
    y_dict = {row[0]:row[1] for row in reader}

    y_ = []
    print '- Regression:: START:: generate training dataset'
    for key in self.dict_to_morph:
      if key in y_dict:
        token_list = self.dict_to_morph[key]
        listing_dict[key] = np.sum([self.w2v[token] for token in token_list if token in self.w2v.vocab.keys()],axis=0)
        y_.append(y_dict[key])
    print '- Regression:: FINISH:: generate training dataset'


    x_ = preprocessing.scale(np.array([listing_dict[key] for key in listing_dict]))
    y_ = np.array(y_)

    self.save_csv('./dataset.csv',x_,y_)

    dat = pd.read_csv('./dataset.csv')
    y_ = dat.ix[:,1].as_matrix()
    x_ = dat.ix[:,2:].as_matrix()

    x_train,x_val,y_train,y_val = train_test_split(x_,y_,test_size=0.1,random_state=0)
    regr = linear_model.Lasso(alpha=10)
    print '- Regression:: START:: train'
    regr.fit(x_train,y_train)
    print '- Regression:: FINISH:: train'
    pred = regr.predict(x_val)
    joblib.dump(regr,save_path,compress=9)


    # print roc_auc_score(y_val,pred)
    print 'FINISH:: Regression'


  def save_csv(self,path, x,y):
    pd.DataFrame(np.c_[y,x]).to_csv(path)

  def predict_cv(self,regr,w2v,tokens):
    x_ = np.sum([w2v[token] for token in tokens if token in w2v.vocab.keys()],axis=0)
    return regr.predict(x_)


  def predict_topic(self, lda_model, token, need_make_bow = False, dictionary=None, n_topics = 5):
    #if token given list
    print token
    if need_make_bow:
      if dictionary is None:
        print 'you need arguments of dictionary'
        return None
      token = dictionary.doc2bow(token)


#    if not isinstance(token,list) or not isinstance(token,dict):
#      print 'token expects list or dictionary but not'
#      return None

    #token given dictionary all patterns
    topic_vec = lda_model[token]
    wc_array = np.empty((len(topic_vec),2), dtype=np.object)
    for i in xrange(len(topic_vec)):
      wc_array[i,0] = topic_vec[i][0]
      wc_array[i,1] = topic_vec[i][1]

    idx = np.argsort(wc_array[:,1])
    idx = idx[::-1]
    wc_array = wc_array[idx]


    topic_str_list = lda_model.print_topic(wc_array[0,0],n_topics)

    if n_topics == 1:
        return [topic_str_list.split('*')[1]]
    else:
        return [t.split('*')[1] for t in topic_str_list.split('+')]


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








