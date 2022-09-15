# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:18:31 2022

@author: bhagy
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, coo_matrix
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import joblib

def to_lower_case(x):
    return x.lower()

def remove_duplicates(df):
    df.drop_duplicates(subset = ['Query', 'Label'], inplace = True)
    df.drop_duplicates(subset = ['Query'], keep = False, inplace = True)
    return df

def single_comments(x):
    return len(re.findall('(--)', x))

def logical_operators(x):
    return len(re.findall("\snot\s|\sand\s|\sor\s|\sxor\s|&&|\|\||!", x))

def find_drop(x):
    return len(re.findall('\sdrop\s', x))

def find_union(x):
    return len(re.findall('\sunion\s', x))

def special_chars(x):
    return len(re.findall("\@|\%|\?|\^", x))

def equal(x):
    return len(re.findall("=", x))

def delimiter(x):
    return len(re.findall(';', x))

def dot(x):
    return len(re.findall('.', x))

def arithmetic_operators(x):
    return len(re.findall("\+|-|\\|\*", x))

def whitespaces(x):
    return len(re.findall("\s", x))

def digits(x):
    return len(re.findall("[0-9]", x))

def alphabets(x):
    return len(re.findall("[a-z]", x))

def brackets(x):
    return len(re.findall("\(|\)|\[|\]", x))

def nulls(x):
    return len(re.findall("null", x))

def hexadecimal(x):
    return len(re.findall("0[xX][0-9a-f]+", x))

def commas(x):
    return len(re.findall(",", x))

def single_quotes_with_gaps(x):
    return len(re.findall("\'", x))

def double_quotes_with_gaps(x):
    return len(re.findall('\"', x))

def cleaning(df):
    df['Query'] = df['Query'].apply(to_lower_case)
    df = remove_duplicates(df)
    return df

def preprocess_data(x):
    features = {}
    features['single_comments'] = single_comments(x)
    features['logical_operators'] = logical_operators(x)
    features['arithmetic_operators'] = arithmetic_operators(x)
    features['drop'] = find_drop(x)
    features['union'] = find_union(x)
    features['special_chars']  = special_chars(x)
    features['equal']  = equal(x)
    features['delimiter'] = delimiter(x)
    features['dot'] = dot(x)
    features['whitespaces'] = whitespaces(x)
    features['digits'] = digits(x)
    features['alphabets'] = alphabets(x)
    features['brackets'] = brackets(x)
    features['nulls'] = nulls(x)
    features['hexadecimal'] = hexadecimal(x)
    features['commas'] = commas(x)
    features['single_quotes'] = single_quotes_with_gaps(x)
    features['double_quotes'] = double_quotes_with_gaps(x)
    
    return features

def predict(string):
    clf = joblib.load('lgbm_model.pkl')
    vectorizer = joblib.load('count_vectorizer.pkl')
    std = joblib.load('std_scaler.pkl')
    string = string.lower()
    test_features = pd.Series(preprocess_data(string))
    bow_features = vectorizer.transform([string])
    test = coo_matrix(hstack((test_features.astype(float), bow_features.astype(float)))).tocsr()
    test = std.transform(test)
    pred = clf.predict(test)
    print(pred[0])
    if pred[0]==0:
        prediction = "Normal SQL Query"
    else:
        prediction = "SQL Injection Query"
    return prediction

#######################################################################

train_data = pd.read_csv('train.csv')
train_data = cleaning(train_data)
train_data = pd.concat([train_data,train_data['Query'].apply(preprocess_data).apply(pd.Series)], axis = 1)

vectorizer = CountVectorizer(ngram_range = (1, 2), min_df = 5)
train_bow = vectorizer.fit_transform(train_data['Query'])
joblib.dump(vectorizer, 'count_vectorizer.pkl')

y_train = train_data['Label']
train_features = train_data.drop(['Query', 'Label'], axis = 1)
train_data_1 = coo_matrix(hstack((train_features, train_bow))).tocsr()

std = StandardScaler(with_mean = False) #Cannot center sparse matrices: pass `with_mean=False` instead.
train_data_1 = std.fit_transform(train_data_1)
joblib.dump(std, 'std_scaler.pkl')

clf = LGBMClassifier(max_depth = 12).fit(train_data_1, y_train)
joblib.dump(clf, 'lgbm_model.pkl')
