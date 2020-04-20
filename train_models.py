import pandas as pd
import numpy as np
import json
import os
import re
import pickle
import utils
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix

import sys
sys.path.insert(0, os.path.dirname(__file__))

# def validate_input(state, text, existing_labels=None):
#     regex = re.compile('[@!#$%^&*()<>?/\|}{~:\s]')

#     if regex.search(text) == None and text != '' and text not in state['RESERVED_KEYS']:
#         if existing_labels:
#             if text not in existing_labels:
#                 return True
#             else:
#                 return False
#         else:
#             return True
    
#     return False

def load_existing_labels():
    existing_labels = None

    with open(os.path.join(os.path.dirname(__file__), 'labels'), 'r') as f:
        existing_labels = f.read().splitlines()

    return {label.split()[0] for label in existing_labels}

def load_label(state):
    existing_labels = load_existing_labels()

    while state['LABEL'] == None:
        label = input('Which label would you like to train?\n')

        if label in existing_labels:
            state['LABEL'] = label
        else:
            print('Please enter a valid label from the following:')
            print(existing_labels)
            # print(', '.join(existing_labels.keys())+'\n')

    return state

def save_data(state):
    filename = input('Filename to save data? \n')

    if filename is '':
        filename = state['DATAFILE']
    if os.path.isfile('data/'+filename):
        overwrite = input("overwrite? any character for undo")
        if overwrite != '':
            return

    print("Saving data\n")
    with open('data/'+filename, 'w') as fout:
        for dic in state['DATA']:
            json.dump(dic, fout)
            fout.write('\n')

def load_labeled_data(state):
    state['DATA'] = utils.load_data(os.path.join(os.path.dirname(__file__), 'data/questions.json'), state['LABEL'])
    return state

def train_model(state):
    train_data, test_data = train_test_split(state['DATA'])
    state['TRAIN'] = train_data
    state['TEST'] = test_data
    X, vectorizer = utils.preprocess_data(train_data[:,0], preprocessor=state['PREPROCESSORS'][state['LABEL']])
    y = train_data[:,1]
    clf = RandomForestClassifier()
    clf.fit(X, y)

    
    state['CLF'] = clf
    state['VECTORIZER'] = vectorizer
    
    return state

def eval_errors(state):
    X, _ = utils.preprocess_data(state['TRAIN'][:,0], state['VECTORIZER'])
    X_test, _ = utils.preprocess_data(state['TEST'][:,0], state['VECTORIZER'])
    y_train = state['TRAIN'][:,1]
    y_test = state['TEST'][:,1]
    train_preds = state['CLF'].predict(X)
    test_preds = state['CLF'].predict(X_test)

    print("Training Error: {}".format(sum(train_preds == y_train) / len(y_train)))
    print("Test Error: {}\n\n".format(sum(test_preds == y_test) / len(y_test)))

def misclassified_questions(state):
    X_test, _ = utils.preprocess_data(state['TEST'][:,0], state['VECTORIZER'])
    y_test = state['TEST'][:,1]
    preds = state['CLF'].predict(X_test)

    for i in range(len(preds)):
        if preds[i] != y_test[i]:
            print("Question: {}, Prediction: {}, Actual: {}".format(state['TEST'][:,0][i], state['CLF'].predict_proba([X_test[i]])[0][1], y_test[i]))

    print("\n\n")

def confusion_matrix(state):
    X_test, _ = utils.preprocess_data(state['TEST'][:,0], state['VECTORIZER'])
    y_test = state['TEST'][:,1]
    preds = state['CLF'].predict(X_test)

    print(state['CLF'].classes_)
    print(multilabel_confusion_matrix(y_test, preds))
    print("\n")

def save_model(state):
    val = input('Overwriting Model... enter x to cancel otherwise press anything\n')
    if val != 'x':
        pickle.dump((state['CLF'], state['VECTORIZER']), open(os.path.join(os.path.dirname(__file__), 'models/'+state['LABEL']+'_clf.pkl'), 'wb'))
        print('\nModel Saved')

def parameter_tune(state):
    n_estimators = [100, 250, 500, 750, 1000]
    max_features = ['auto', 'log2']
    max_depth = [None, 100, 500, 1000]
    min_samples_split = [2, 5, 10, 12]
    min_samples_leaf = [1, 2, 4, 8]
    criterion = ['gini', 'entropy']
    bootstrap = [True, False]

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'criterion': criterion,
        'bootstrap': bootstrap
    }

    X, _ = utils.preprocess_data(state['TRAIN'][:,0], state['VECTORIZER'])
    y = state['TRAIN'][:,1]
    print('Tuning Model Parameters...\n\n')
    clf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_jobs=-1)
    rf_random.fit(X, y)
    state['CLF'] = rf_random.best_estimator_
    return state

def main():
    preprocessors = {
        'actual_question': utils.actual_question_preprocessor,
        # 'question_type': utils.question_type_preprocessor
    }
    state = {'LABEL': None, 'DATA': None, 'TRAIN': None, 'TEST': None, 
    'CLF': None, 'VECTORIZER': None, 'PREPROCESSORS': preprocessors}

    state = load_label(state)
    state = load_labeled_data(state)
    state = train_model(state)
    eval_errors(state)

    while (True):
        print('----- s for save --------- c for confusion matrix --------- ')
        print('------ m for misclassified questions ------- e for errors -------')
        val = input('------ p for parameter tuning -------- r for retrain --------- q for quit ----------\n')

        if val == 'r':
            state = train_model(state)
            eval_errors(state)
        elif val == 's':
            save_model(state)
        elif val == 'c':
            confusion_matrix(state)
        elif val == 'm':
            misclassified_questions(state)
        elif val == 'e':
            eval_errors(state)
        elif val == 'p':
            state = parameter_tune(state)
            eval_errors(state)
        elif val == 'q':
            exit()


if __name__ == "__main__":
    main()