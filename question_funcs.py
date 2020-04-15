import pickle
import utils

def is_actual_question(question, threshold=.4):
	clf, vectorizer = pickle.load(open('models/actual_question_clf.pkl', 'rb'))
	question, _ = utils.preprocess_data([question], vectorizer)
	return clf.predict_proba(question)[0][1] >= threshold
