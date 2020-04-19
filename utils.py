import numpy as np
from word2number import w2n
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# def filter_dataset(filename):
#     questions = list(set(pd.read_json(filename, lines=True)["question"]))
#     questions.sort(key=lambda x: len(x), reverse=True)
    
#     filtered = []
    
#     for question in questions:
#         if not any([question in filtered_question["question"] for filtered_question in filtered]):
#             filtered.append({"question": question})

#     np.random.shuffle(filtered)

#     with open('/Users/michaelli/Desktop/data_labeler/data/questions.json', 'w') as fout:
#         for dic in filtered:
#             json.dump(dic, fout)
#             fout.write("\n")
    
#     return filtered

def load_data(filename, label_name):
    with open(filename, 'r') as f:
        return np.array([[eval(data_point)['question'], eval(data_point)[label_name]] for data_point in f.read().splitlines() if label_name in eval(data_point)])

def remove_non_ascii(question):
	return ''.join([i if ord(i) < 128 else '' for i in question])

def is_filename(text):
    if '.c' in text or '.py' in text:
        return True
    else:
        return False

def is_spelled_out_number(text):
    try:
        w2n.word_to_num(text)
        return True
    except ValueError:
        return False

def is_TA_or_instructor_name(text):
    names = {"adam", "victor", "jesse", "peter"}
    
    if text in names:
        return True
    else:
        return False

def actual_question_preprocessor(sentence):
	sentence = remove_non_ascii(sentence)
	tokenizer = RegexpTokenizer("[^;\s.?,!()]+\.c|[^;\s.,?!()]+\.py|[^;\s.?!(),]+\(\)|[^;\s.?,!()]+")
	sentence = tokenizer.tokenize(sentence.lower())

	for i in range(len(sentence)):
		if sentence[i].isnumeric():
			sentence[i] = "numericnumber"
		elif is_spelled_out_number(sentence[i]):
			sentence[i] = "nonnumericnumber"
		elif is_filename(sentence[i]):
			sentence[i] = "filename"
		elif is_TA_or_instructor_name(sentence[i]):
			sentence[i] = "name"
		# elif is_function(sentence[i]):
		# 	sentence[i] = "function"
		# elif is_snake_case(sentence[i]):
		# 	sentence[i] = "snakecase"

	return ' '.join(sentence)

def preprocess_data(x_data, vectorizer=None, preprocessor=None):
	sentence_lens = [[len(sentence.split())] for sentence in x_data]
	X = None

	if vectorizer:
		X = vectorizer.transform(x_data)
	else:
		vectorizer = CountVectorizer(preprocessor=preprocessor)
		# vectorizer = TfidfVectorizer(preprocessor=preprocess_sentence)
		X = vectorizer.fit_transform(x_data)

	X = np.hstack((X.toarray(), sentence_lens))

	return X, vectorizer
