import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import utils

def is_actual_question(question, threshold=.4):
	clf, vectorizer = pickle.load(open(os.path.join(os.path.dirname(__file__), 'models/actual_question_clf.pkl'), 'rb'))
	question, _ = utils.preprocess_data([question], vectorizer)
	return clf.predict_proba(question)[0][1] >= threshold
