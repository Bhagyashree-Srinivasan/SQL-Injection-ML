# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 20:31:12 2022

@author: bhagy
"""

from flask import Flask, jsonify, request
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, coo_matrix
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import joblib

import flask
app = Flask(__name__)

###############################################################################

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

#####################################################################################

@app.route('/')
def hello_world():
    return 'Find out whether a query is an injection query or not.'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('lgbm_model.pkl')
    vectorizer = joblib.load('count_vectorizer.pkl')
    std = joblib.load('std_scaler.pkl')
    to_predict_list = request.form.to_dict()
    string = to_predict_list['query_text'].lower()
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

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)



