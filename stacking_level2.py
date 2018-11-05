# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:24:05 2018

"""

import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
import gc
import matplotlib.pyplot as plt

## Add OOF predictions
train_base = pd.read_csv('./model_1/train_predictions_model1.csv', header=None)
test_base = pd.read_csv('./model_1/test_predictions_model1.csv', header=None)

train_base.columns = train_base.columns.map(str)
train_base.columns = 'model_1_' + train_base.columns
train_base = train_base.rename(columns={train_base.columns[0]:'fname'})
train_base = train_base.sort_values(by=['fname'])

test_base.columns = test_base.columns.map(str)
test_base.columns = 'model_1_' + test_base.columns
test_base = test_base.rename(columns={test_base.columns[0]:'fname'})
test_base = test_base.sort_values(by=['fname'])
gc.collect()


## Add statistical features
train_stats = pd.read_csv('./stats/train_stat.csv')
train_stats.columns = train_stats.columns.map(str)
train_stats.columns = 'stats_' + train_stats.columns
train_stats = train_stats.rename(columns={'stats_fname':'fname'})
train_stats = train_stats.sort_values(by=['fname'])
gc.collect()

test_stats = pd.read_csv('./stats/test_stat.csv')
test_stats.columns = test_stats.columns.map(str)
test_stats.columns = 'stats_' + test_stats.columns
test_stats = test_stats.rename(columns={'stats_fname':'fname'})
test_stats = test_stats.sort_values(by=['fname'])
gc.collect()

train = pd.merge(train_base, train_stats, on='fname')
test = pd.merge(test_base, test_stats, on='fname')

train_index = pd.read_csv('../input/train.csv')
submission = pd.read_csv('../input/sample_submission.csv')

weights_train=np.ones(train_index.shape[0])
weights_train[train_index.manually_verified==0]=0.6
weights_test=np.ones(submission.shape[0])

n_categories = len(train_index.label.unique())
print("Number of unique categories: {}".format(n_categories))

train = pd.merge(train_index, train ,on='fname')
test = pd.merge(submission, test, on='fname')

feature_names = list(test.drop(['fname', 'label', 'stats_len'], axis=1).columns.values)


NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
kf = kfold.split(train[feature_names], train['label'])
cv_train = np.zeros([len(train['label']),n_categories])
cv_pred = np.zeros([test.shape[0],n_categories])
best_trees = []
fold_scores = []


X = train[feature_names]
X_test = test[feature_names]
le = preprocessing.LabelEncoder()
le.fit(train['label'])
train_label = le.transform(train['label'])

cv_train_lgb = np.zeros([len(train['label']),n_categories])
cv_pred_lgb= np.zeros([test.shape[0],n_categories])
params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': 5,
    'num_leaves': 31,
    'learning_rate': 0.025,
    'feature_fraction': 0.85,
    'lambda_l2': 1.5,
    'num_class': n_categories,
}

for i, (train_fold, validate) in enumerate(kf):
    print('Fold {}/{}'.format(i + 1, 5))
    X_train, X_validate, label_train, label_validate = \
                X.iloc[train_fold, :], X.iloc[validate, :], train_label[train_fold], train_label[validate]
    lgb_train = lgb.Dataset(X_train, label_train, feature_name=feature_names, weight=weights_train[train_fold])
    lgb_valid = lgb.Dataset(X_validate, label_validate, feature_name=feature_names, weight=weights_train[validate])
    lgb_test = lgb.Dataset(X_test, feature_name=feature_names,weight=weights_test)

    bst = lgb.train(
        params_lgb,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100,
        verbose_eval=50,
    )

    best_trees.append(bst.best_iteration)
    #ax = lgb.plot_importance(bst, max_num_features=10, grid=False, height=0.8, figsize=(16, 8))
    #plt.show()

    cv_pred_lgb += bst.predict(X_test)
    cv_train_lgb[validate] += bst.predict(X_validate)
    score = accuracy_score(np.argmax(cv_train_lgb[validate],axis= 1),label_validate)
    print(score)
    fold_scores.append(score)


top_3 = np.argsort(-cv_pred, axis=1)[:, :3]
predicted_labels = [' '.join(list(le.inverse_transform(x))) for x in top_3]
submission['label'] = predicted_labels
submission.to_csv('final_answer.csv',index=False)

