import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

def train_actual_question():
    actual_question_data = utils.load_data(os.path.join(os.path.dirname(__file__), 'data/questions.json'), 'actual_question')

    train_data, test_data = train_test_split(actual_question_data)
    X, vectorizer = utils.preprocess_data(train_data[:,0], preprocessor=utils.actual_question_preprocessor)
    y = train_data[:,1]
    clf = RandomForestClassifier()
    clf.fit(X, y)

    X_test, _ = utils.preprocess_data(test_data[:,0], vectorizer)
    y_test = test_data[:,1]
    y_preds = clf.predict(X_test)
    # print("Cross Val Score: {}".format(cross_val_score(clf, X_test, y_test).mean()))
    print("Training Error: {}".format(sum(clf.predict(X) == y) / len(y)))
    print("Test Error: {}".format(sum(y_preds == y_test) / len(y_preds)))
    print(clf.classes_)
    print(multilabel_confusion_matrix(y_test, y_preds))
    print("\n")
    print("Misclassified Questions")
    for i in range(len(y_preds)):
        if y_preds[i] != y_test[i]:
            print("Question: {}, Prediction: {}, Actual: {}".format(test_data[:,0][i], clf.predict_proba([X_test[i]])[0][1], y_test[i]))

    pickle.dump((clf, vectorizer), open(os.path.join(os.path.dirname(__file__), 'models/actual_question_clf.pkl'), 'wb'))
    print('\nModel Saved')