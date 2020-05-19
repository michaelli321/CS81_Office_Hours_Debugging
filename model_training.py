import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
# from sklearn.preprocessing import StandardScaler

class WordCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self, X, y=None):
        return [[len(sentence.split())] for sentence in X]
    
    def fit(self, X, y=None):
        return self
    
class CharacterCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self, X, y=None):
        return [[len(sentence)] for sentence in X]
    
    def fit(self, X, y=None):
        return self
    
class StopWordFrequency(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor
    # check this logic
    def transform(self, X, y=None):
        transformed_x = [self.preprocessor(sentence) for sentence in X]
        return [[(len(X[i])-len(transformed_x[i]))/len(X[i])] if len(X[i]) != 0 else [0] for i in range(len(X))]
    
    def fit(self, X, y=None):
        return self

def train_actual_question():
    actual_question_data = utils.load_data(os.path.join(os.path.dirname(__file__), 'data/questions.json'), 'actual_question')

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngram', CountVectorizer(preprocessor=utils.actual_question_preprocessor)),
            ('word_count', WordCountTransformer()),
            ('character_count', CharacterCountTransformer()),
            ('setnence_remain', StopWordFrequency(preprocessor=utils.actual_question_preprocessor))
        ])),
        ('clf', RandomForestClassifier())
    ])

    X, y = actual_question_data[:,0], actual_question_data[:,1]
    pipeline.fit(X, y)

    return pipeline

def train_answerable():
    answerable_data = utils.load_data(os.path.join(os.path.dirname(__file__), 'data/tot.json'), 'answerable')
    filtered_data = np.array([[answerable_data[:,0][i], 't'] if (answerable_data[:,1][i] == 't' or 
        answerable_data[:,1][i] == 'c') else [answerable_data[:,0][i], 'f'] for i in range(len(answerable_data))])

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('vectorizer', TfidfVectorizer(preprocessor=utils.answerable_preprocessor)),
            ('char_count', CharacterCountTransformer()),
            ('word_count', WordCountTransformer())
        ])),
        ('clf', LogisticRegression(max_iter=1000, class_weight={'t':2.5}))
    ])

    X, y = filtered_data[:,0], filtered_data[:,1]
    pipeline.fit(X, y)

    return pipeline
