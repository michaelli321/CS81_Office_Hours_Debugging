import pandas as pd
import numpy as np
import json
import os
import re
import pickle
import utils

import sys
sys.path.insert(0, os.path.dirname(__file__))

# def filter_dataset(filename):
#     questions = list(set(pd.read_json(filename, lines=True)["question"]))
#     questions.sort(key=lambda x: len(x), reverse=True)
    
#     filtered = []
    
#     for question in questions:
#         if not any([question in filtered_question["question"] for filtered_question in filtered]):
#             filtered.append({"question": question})

#     np.random.shuffle(filtered)

#     with open('questions.json', 'w') as fout:
#         for dic in q:
#             json.dump(dic, fout)
#             fout.write("\n")
    
#     return filtered

def load_data(state):
    print('Loading Data...\n')
    with open('data/'+state['DATAFILE'], 'r') as f:
        state['DATA'] = [eval(data_point) for data_point in f.read().splitlines()]

    return state

def prompt_and_load_file(state):
    while state['DATA'] == None:
        filename = input('Name of dataset to load?\n')
        path = 'data/'+filename

        if filename is '' or os.path.isfile(path):
            state['DATAFILE'] = filename if filename is not '' else state['DATAFILE']
            state = load_data(state)
        else:
            print('Please enter a valid filename\n')

    return state

def validate_input(state, text, existing_labels=None):
    regex = re.compile('[@!#$%^&*()<>?/\|}{~:\s]')

    if regex.search(text) == None and text != '' and text not in state['RESERVED_KEYS']:
        if existing_labels:
            if text not in existing_labels:
                return True
            else:
                return False
        else:
            return True
    
    return False

def load_existing_labels():
    existing_labels = None

    with open('labels', 'r') as f:
        existing_labels = f.read().splitlines()

    return {label.split()[0]: set(label.split()[1:]) for label in existing_labels}

def get_values_for_new_label(state):
    values = []
    value = input("Please enter a value for the label\n")
    
    while value != '':
        if validate_input(state, value):
            values.append(value)
        else:
            print("Please enter a valid value (No special characters)")

        value = input("Please enter another value for the label\n")

    return values

def create_new_label(state, existing_labels):
    label = input('What is the name of the new label?\n')

    while not validate_input(state, label, existing_labels):
        label = input("Please enter a valid new label\n")

    values = get_values_for_new_label(state)
    
    with open('labels', 'a') as f:
        f.write(label + ' '+' '.join(values)+'\n')

    state['LABEL'] = label
    state['LABEL_VALS'] = values

    return state

def load_classifier(state):
    filename = os.path.join(os.path.dirname(__file__), 'models/'+state['LABEL']+'_clf.pkl')
    if os.path.exists(filename):
        state['CLF'] = pickle.load(open(filename, 'rb'))

    return state

def load_label(state):
    existing_labels = load_existing_labels()

    while state['LABEL'] == None:
        label = input('type label or n for new label\n')

        if label == 'n':
            state = create_new_label(state, existing_labels)
        else:
            if label in existing_labels:
                state['LABEL'] = label
                state['LABEL_VALS'] = existing_labels[label]
                state = load_classifier(state)
            else:
                print('Please enter a valid label from the following:')
                print(', '.join(existing_labels.keys())+'\n')

    return state

def get_label_stats(state):
    labeled = 0
    val_splits = {val: 0 for val in state['LABEL_VALS']}
    # print(state['DATA'][:10])
    for data_point in state['DATA']:
        if state['LABEL'] in data_point:
            labeled += 1
            # print(data_point)
            val_splits[data_point[state['LABEL']]] += 1

    return labeled, val_splits

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

def uncertainty(state, question):
    model = state['CLF']
    probability = model.predict_proba([question])[0][1]
    confidence = abs(probability - .5)
    return confidence

def main():
    state = {'LABEL': None, 'DATA': None, 'DATAFILE': 'questions.json', 'LABEL_VALS': None, 
    'RESERVED_KEYS': {'s', 'n', 'l', 'q'}, 'CLF': None}

    # eventually want to have option to filter data and append to existing dataset
    state = prompt_and_load_file(state)
    state = load_label(state)

    num_labeled, val_splits = get_label_stats(state)

    if state['CLF']:
        state['DATA'] = sorted(state['DATA'], key=lambda x: uncertainty(state, x['question']))
    else:
        np.random.shuffle(state['DATA'])

    for i in range(len(state['DATA'])):
        if state['LABEL'] in state['DATA'][i]:
            continue
        else:
            while True:
                print('\n\n\n-------' + state['LABEL'] + '---------Labeled: ' + 
                    str(num_labeled) + '--------To Do: ' + 
                    str(len(state['DATA'])-num_labeled) + '---------')
                print('==========++++++++++*********++++++++++==========')
                print(state['DATA'][i]['question'])
                print('==========++++++++++*********++++++++++==========')
                val = input('------ s for save -------- n for next -------- ' +
                    'l for label stats --------- q for quit ----------\n')

                if val in state['LABEL_VALS']:
                    state['DATA'][i][state['LABEL']] = val
                    num_labeled += 1
                    val_splits[val] += 1
                    break
                elif val == 'n':
                    break
                elif val == 'l':
                    print(val_splits)
                elif val == 's':
                    save_data(state)
                elif val == 'q':
                    exit()
                else:
                    print('Please enter a valid value\n')

if __name__ == "__main__":
    main()