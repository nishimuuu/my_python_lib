# -*- coding:utf-8 -*-

__author__ = 'takahiro'

import sys
import cPickle as pickle

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale


class Modeler:
  # dataset must be pandas::DataFrame
  def __init__(self, x_dataset, y_dataset):
    self.x_dataset = x_dataset
    self.y_dataset = y_dataset

  def preprocess(self,
                 seed        = 246,
                 y_idx       = 0,
                 method      = 'svm',
                 gridsearch  = True,
                 scaling     = True,
                 train_ratio = 0.8):
    self.seed = seed
    self.method = method
    self.gridsearch = gridsearch
    self.train_ratio = train_ratio
    if y_idx < 0 or self.dataset.shape[0] < 2:
      print 'invalid y_index or dataset dimention: cannot train the data'
      sys.exit(1)

    if scaling:
      self.x_dataset = scale(self.x_dataset)

    self.x_train, \
    self.x_test, \
    self.y_train, \
    self.y_test = train_test_split(self.x_dataset,
                                   self.y_dataset,
                                   test_size = 1-train_ratio,
                                   random_state = seed )

  def train(self):
    if   self.method == 'svm':
      from sklearn.svm import SVC
      classifier = SVC()
      grid_params = {'kernel' : ('rbf'),
                     'C'      : np.logspace(-4, -4, 10),
                     'gamma'  : np.logspace(-4, -4, 10)
      }

    elif self.method == 'randomforest':
      from sklearn.ensemble import RandomForestClassifier
      classifier = RandomForestClassifier()
      grid_params = {'n_estimators'      : np.arrange(5, 300, 10),
                     'max_features'      : np.arrange(5, 20, 5),
                     'random_state'      : 0,
                     'n_jobs'            : 1,
                     'min_samples_split' : np.arrange(5, 100, 5),
                     'max_depth'         : np.arrange(5, 100, 5)
      }

    elif self.method == 'gradientboosting':
      from sklearn.ensemble import GradientBoostingClassifier
      classifier = GradientBoostingClassifier()
      grid_params = {'learning_rate'     : np.arrange(0.01, 0.9, 0.05),
                     'n_estimators'      : np.arrange(10, 100, 10),
                     'max_depth'         : np.arrange(1, 3, 1),
                     'min_samples_split' : np.arrange(1, 10, 1),
                     'min_samples_leaf'  : np.arrange(1, 10, 1)
      }
    else:
      import xgboost as xgb
      data = xgb.DMatrix(self.x_train, label=self.y_train)
      classifier = xgb.train(data, early_stopping_rounds=10)

    if self.method in ['svm', 'randomforest', 'gradientboosting']:
      if self.gridsearch:
        classifier = GridSearchCV(classifier,grid_params)
      classifier.fit(self.x_train, self.y_train)

    self.model = classifier

  def evaluate(self, metrics):
    print 'evaluate'

  def get_model(self):
    return self.model

  def predict(self, array):
    return self.model.predict(np.array(array))

  def save_model(self,path):
    self.y_variable  = None
    self.x_variables = None
    self.x_train     = None
    self.x_test      = None
    self.y_train     = None
    self.y_test      = None
    pickle.dump(self, open(path),'w')

