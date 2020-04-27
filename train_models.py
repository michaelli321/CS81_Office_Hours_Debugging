import numpy as np
import os
import pickle
import utils
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix
import model_training

import sys
sys.path.insert(0, os.path.dirname(__file__))

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

    return state

def load_labeled_data(state):
    state['DATA'] = utils.load_data(os.path.join(os.path.dirname(__file__), 'data/questions.json'), state['LABEL'])
    return state

def eval_errors(state):
    X, y = state['DATA'][:,0], state['DATA'][:,1]
    preds = state['MODEL'].predict(X)

    print("Training Error: {}".format(sum(preds == y) / len(y)))
    print("Cross Val Score: {}".format(cross_val_score(state['MODEL'], X, y).mean()))

def misclassified_questions(state):
    X, y = state['DATA'][:,0], state['DATA'][:,1]
    preds = state['MODEL'].predict(X)

    for i in range(len(preds)):
        if preds[i] != y[i]:
            print("Question: {}, Prediction: {}, Actual: {}".format(X[i], state['MODEL'].predict_proba([X[i]])[0][1], y[i]))

    print("\n\n")

def confusion_matrix(state):
    X, y = state['DATA'][:,0], state['DATA'][:,1]
    preds = state['MODEL'].predict(X)

    print(state['MODEL']['clf'].classes_)
    print(multilabel_confusion_matrix(y, preds))
    print("\n")

def save_model(state):
    val = input('Overwriting Model... enter x to cancel otherwise press anything\n')
    if val != 'x':
        pickle.dump(state['MODEL'], open(os.path.join(os.path.dirname(__file__), 'models/'+state['LABEL']+'_clf.pkl'), 'wb'))
        print('\nModel Saved')

def parameter_tune(state):
    param_grid = {
        'features__ngram__ngram_range': [(1,1), (1,2), (1,3)],
        'features__ngram__max_features': [None, 100, 250, 500, 1000, 1500, 2000, 2500, 3000],
        'clf__n_estimators': [100, 250, 500, 750, 1000],
        'clf__max_features': ['auto', 'log2'],
        'clf__max_depth': [None, 100, 500, 1000],
        'clf__min_samples_split': [2, 5, 10, 12],
        'clf__min_samples_leaf': [1, 2, 4, 8],
        'clf__criterion': ['gini', 'entropy'],
        'clf__bootstrap': [True, False]
    }

    search = RandomizedSearchCV(state['MODEL'], param_grid, n_jobs=-1, n_iter=500)
    search.fit(state['DATA'][:,0], state['DATA'][:,1])
    state['MODEL'] = search.best_estimator_
    return state


def main():
    model_training_funcs = {
        'actual_question': model_training.train_actual_question,
    }

    state = {'LABEL': None, 'DATA': None, 'MODEL': None}

    state = load_label(state)
    state = load_labeled_data(state)
    state['MODEL'] = model_training_funcs[state['LABEL']]()
    eval_errors(state)

    while (True):
        print('----- s for save --------- c for confusion matrix --------- ')
        print('------ m for misclassified questions ------- e for errors -------')
        val = input('------ p for parameter tuning --------- q for quit ----------\n')

        if val == 's':
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