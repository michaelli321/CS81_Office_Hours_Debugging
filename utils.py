import numpy as np
from word2number import w2n
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words('English'))
stop_words.update({
    'help',
    'please',
    'need',
    'problem',
    'stuff',
    'thing',
    'question',
    'work',
    'working',
    'works'
    })

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
        return np.array([[remove_non_ascii(eval(data_point)['question']), eval(data_point)[label_name]] for data_point in f.read().splitlines() if label_name in eval(data_point)])

def remove_non_ascii(question):
    return ''.join([i if ord(i) < 128 else '' for i in question])

def is_filename(text):
    file_extension = {'.c', '.py', '.h', '.txt'}

    for ext in file_extension:
        if ext in text:
            return True
    return False

def is_spelled_out_number(text):
    try:
        w2n.word_to_num(text)
        return True
    except ValueError:
        return False

def is_TA_or_instructor_name(text):
    names = {"adam", "victor", "jesse", "peter"}
    
    for name in names:
        if name in text:
            return True

    return False

def check_quotations(text):
    text = text.replace('"', "")
    text = text.replace("'", "")
    text = text.replace("`", "")
    return text

def is_error_code(text):
    errors = {
    'seg',
    'overflow',
    'assertion',
    'nullpointer',
    'illegal',
    'timeout',
    'exception',
    }

    for err in errors:
        if err in text:
            return True

    return False

def is_system_word(text):
    system_words = {'git', 'ssh', 'intellij', 'server', 'sdl', 'sdk', 'config', 'makefile', 'vm'}
    
    for word in system_words:
        if word in text:
            return True
    
    return False

def is_function(text):
    if '()' in text:
        return True
    else:
        return False

def check_backslash(text):
    if "\\" in text:
        return text

    return re.sub(r"^[a-zA-Z0-9]+\/[a-zA-z0-9]+\Z", ' '.join(text.split('/')), text)

def check_camelcase(text):
    if 'git' not in text.lower() and 'exception' not in text.lower() and re.match(r"^[a-zA-Z]+([A-Z][a-z0-9]+)+", text):
        return True
    else:
        return False

def check_snake_case(text):
    if re.match(r"[a-zA-Z_]+_[a-zA-Z_]+", text):
        return True
    else:
        return False

def check_camel_snakecase(text):
    if check_camelcase(text) or check_snake_case(text):
        return 'camelsnakecase'
    else:
        return text
    
def actual_question_preprocessor(sentence):
    sentence = remove_non_ascii(sentence)
    tokenizer = RegexpTokenizer("[^;\s.?,!()]+\.c|[^;\s.,?!()]+\.py|[^;\s.,?!()]+\.h|[^;\s.,?!()]+\.txt|[^;\s.?!(),]+\(\)|[^;\s.?,!()]+")
    sentence = tokenizer.tokenize(sentence)

    for i in range(len(sentence)):
        sentence[i] = check_quotations(sentence[i])
        sentence[i] = check_backslash(sentence[i])
        sentence[i] = check_camel_snakecase(sentence[i])
        sentence[i] = sentence[i].lower()

        if sentence[i] in stop_words:
            sentence[i] = ""
        elif sentence[i].isnumeric():
            sentence[i] = "numericnumber"
        elif is_spelled_out_number(sentence[i]):
            sentence[i] = "nonnumericnumber"
        elif is_filename(sentence[i]):
            sentence[i] = "filename"
        elif is_TA_or_instructor_name(sentence[i]):
            sentence[i] = "name"
        elif is_error_code(sentence[i]):
            sentence[i] = "errorcode"
        elif is_system_word(sentence[i]):
            sentence[i] = "sys"
        elif is_function(sentence[i]):
          sentence[i] = "func"
        elif sentence[i] in stopwords.words('english'):
            sentence[i] = ""

    return ' '.join(sentence)
