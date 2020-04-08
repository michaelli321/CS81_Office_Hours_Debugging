import pandas as pd
import numpy as np
import os

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

def load_file(state):
    path = input('Name of dataset to load?\n')

    if path is '' or os.path.isfile(path):
        state['DATAFILE'] = path if path is not "" else state['DATAFILE']
        state = load_data(state)
    else:
        print('Please enter a valid filename\n')

    return state

def main():
    state = {'CURRENT_LABEL': None, 'DATA': None, 'DATAFILE': 'data/questions.json'}

    # eventually want to have option to filter data and append to existing dataset
    while state['DATA'] == None:
        state = load_file(state)

    label = input('type label or n for new label')




    while True:
        break
    # label data or create new label?
    # if label data -> which label?
    # print how many unlabeled? how many to-do?
    # skip option
    # save dataset -- overwrite or new file name?

if __name__ == "__main__":
    main()