import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import model_training
import pickle
import utils

def is_actual_question(question, threshold=.5):
    model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'models/actual_question_clf.pkl'), 'rb'))
    return model.predict_proba([question])[0][1] >= threshold

def is_answerable(question, threshold=.5):
	model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'models/answerable_clf.pkl'), 'rb'))
	return model.predict_proba([question])[0][1] >= threshold
