import pandas as pd
import numpy as np
import os
import re

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
    state['DATA'] = pd.read_json(state['DATAFILE'], lines=True).to_dict('records')
    return state

def prompt_and_load_file(state, default='data/questions.json'):
    while state['DATA'] == None:
        path = input('Name of dataset to load?\n')

        if path is '' or os.path.isfile(path):
            state['DATAFILE'] = path if path is not "" else state['DATAFILE']
            state = load_data(state)
        else:
            print('Please enter a valid filename\n')

    return state

def validate_label(label, existing_labels):
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:\s]')

    if label not in existing_labels and regex.search(string) == None:
        return True
    else:
        return False

def load_existing_labels():
    existing_labels = None

    with open('labels', 'r') as f:
        existing_labels = f.read().splitlines()

    return set(existing_labels)

def create_new_label(state):
    label = input('What is the name of the new label?\n')

    while not validate_label(label, existing_labels):
        label = input("Please enter a valid new label\n")

    with open(labels, 'a') as f:
        f.write(label+'\n')

    state['LABEL'] = label
    return state

def load_label(state):
    existing_labels = load_existing_labels()

    while state['LABEL'] == None:
        label = input('type label or n for new label\n')

        if label == 'n':
            state = create_new_label(state)
        else:
            if label in existing_labels:
                state['LABEL'] = label
            else:
                print('Please enter a valid label from the following:')
                print(existing_labels)

    return state

def main():
    state = {'LABEL': None, 'DATA': None, 'DATAFILE': 'data/questions.json'}

    # eventually want to have option to filter data and append to existing dataset
    state = prompt_and_load_file(state)
    state = load_label(state)

    while True:
        break
    # label data or create new label?
    # if label data -> which label?
    # print how many unlabeled? how many to-do?
    # skip option
    # save dataset -- overwrite or new file name?

if __name__ == "__main__":
    main()