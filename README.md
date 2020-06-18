## CS81: Undergraduate Projects in Computer Science:

I worked with Professor Adam Blank on a project to help students ask better questions using the online office hours question tool. I developed two models that filtered questions on whether they were an actual question ("What is a Linked List" vs "pls help") and whether or not the question was answerable. I created a data labeling interface `label_data.py` and a model training interface `train_models.py` to aid in this task. `question_funcs.py` holds the functions that return T/F for a given question for the given model. 

CV Accuracy of actual question model: ~91% <br />
CV Accuracy of answerable model: ~85%
