import pandas as pd
import numpy as np

CURRENT_LABEL = None
DATA = None
DATAFILE = 'data/questions.json'

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

def load_data():
    while DATA == None:
        try:
            datafile = input('Name of dataset to load?\n')
            if datafile == '':
                DATA = pd.read_json(DATAFILE, lines=True).to_dict('records')
            else:
                load_data(datafile)
        except ValueError:
            print('Please enter a valid filename\n')

    DATA = pd.read_json(filename, lines=True).to_dict('records')

    print('Loading Data...\n')

def main():
    load_data() # eventually want to have option to filter data and append to existing dataset
    
    

    

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