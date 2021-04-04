from nltk.stem import WordNetLemmatizer
import nltk
import os
from os.path import join
from urllib.request import urlopen
import json
import numpy as np
from tensorflow import keras
import pickle
import random
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


model = load_model('Trained_Models/bot.h5')
intents = json.loads(open('Trained_Models/intents.json').read())
words = pickle.load(open('Trained_Models/words.pkl', 'rb'))
classes = pickle.load(open('Trained_Models/classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.9
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    try:
        if results[0]:
            for r in results:
                return_list.append(
                    {"intent": classes[r[0]], "probability": str(r[1])})
        else:
            return_list.append({"intent": 'noanswer', "probability": '1'})
    except:
        return_list.append({"intent": 'noanswer', "probability": '1'})
    # print(return_list)
    return return_list


def getResponse(ints, intents_json):
    result = 'sorry i could not understand'
    tag = ints[0]['intent']
    # print(tag)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result, tag


def prediction(msg):
    ints = predict_class(msg, model)
    res, tag = getResponse(ints, intents)
    if float(ints[0]['probability']) > 0.95:
        result = {"response": res,
                  "tag": tag
                  }
    else:
        result = {"response": "sorry i could not understand what you are saying",
                  "tag": "couldnotunderstand"
                  }

    return result
# text to response


def to_json(response, tag=None, Notes=None, urls=None,):
    stuf_for_frontend = {
        "response": response,
        "tag": tag,
        "notes": Notes,
        "urls": urls,
    }

    return stuf_for_frontend


# taking command from response
def takeCommand(cmd):

    return cmd


# # predicting solution from machine learning model
# def prediction(query):
#     intents = None
#     with open("Backend/intents.json") as file:
#         data = json.load(file)
#     model = keras.models.load_model('chat_model')
#     with open('Backend/tokenizer.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
#     with open('Backend/label_encoder.pickle', 'rb') as enc:
#         lbl_encoder = pickle.load(enc)
#     max_len = 20
#     result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([query]),
#                                                                       truncating='post', maxlen=max_len))
#     # print(tensorflow.Tensor(result))
#     # print(np.max(result))
#     tag = lbl_encoder.inverse_transform([np.argmax(result)])
#     for i in data['intents']:
#         if i['tag'] == tag:
#             intents = to_json(response=np.random.choice(
#                 i['responses']), tag=str(tag))
#     return intents


def clear(): return os.system('cls')


clear()

# opening intents
with open("Trained_Models/intents.json") as file:
    data = json.load(file)


def converttostring(list):
    res = str(", ".join(map(str, list)))
    return res


##### response #####

def ChatBot(query):

    try:

        Bot_Response = prediction(query)
        tag = Bot_Response.get('tag')
        # open stack overflow
        if tag == 'stackoverflow':

            Bot_Response = to_json(
                response='oh but i cant fix your error right now')

    except Exception as e:
        Bot_Response = to_json('sorry something went wrong')
        # print(e)

    return Bot_Response


while True:
    x = input('user : ')
    print(ChatBot(takeCommand(x).lower()))
