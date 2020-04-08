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
            state['DATAFILE'] = path if path is not '' else state['DATAFILE']
            state = load_data(state)
        else:
            print('Please enter a valid filename\n')

    return state

def validate_input(text, existing_labels=None):
    regex = re.compile('[@!#$%^&*()<>?/\|}{~:\s]')

    if regex.search(text) == None and text != '':
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

def get_values_for_new_label():
    values = []
    value = input("Please enter a value for the label\n")
    
    while value != '':
        if validate_input(value):
            values.append(value)
        else:
            print("Please enter a valid value (No special characters)")

        value = input("Please enter another value for the label\n")

    return values

def create_new_label(state, existing_labels):
    label = input('What is the name of the new label?\n')

    while not validate_input(label, existing_labels):
        label = input("Please enter a valid new label\n")

    values = get_values_for_new_label()
    
    with open('labels', 'a') as f:
        f.write(label + ' '+' '.join(values)+'\n')

    state['LABEL'] = label
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
            else:
                print('Please enter a valid label from the following:')
                print(', '.join(existing_labels.keys())+'\n')

    return state

def get_label_stats(state):
    labeled = 0
    val_splits = {val: 0 for val in state['LABEL_VALS']}

    for data_point in state['DATA']:
        if state['LABEL'] in data_point:
            labeled += 1
            val_splits[data_point[state['LABEL']]] += 1

    return labeled, val_splits

def save_data(state):
    pass
    # save dataset -- overwrite or new file name?

def main():
    state = {'LABEL': None, 'DATA': None, 'DATAFILE': 'data/questions.json', 'LABEL_VALS': None}

    # eventually want to have option to filter data and append to existing dataset
    state = prompt_and_load_file(state)
    state = load_label(state)

    num_labeled, val_splits = get_label_stats(state)

    np.random.shuffle(state['DATA'])

    for i in range(len(state['DATA'])):
        if state['LABEL'] in state['DATA'][i]:
            continue
        else:
            while True:
                print('\n\n\n-------' + state['LABEL'] + '---------Labeled: ' + 
                    str(num_labeled) + '--------To Do: ' + 
                    str(len(state['DATA'])-num_labeled) + '---------')
                print(state['DATA'][i]['question'])
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