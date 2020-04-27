import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

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

