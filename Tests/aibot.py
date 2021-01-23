import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import  WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import reset_uids

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
    
words = pickle.load(open('words.pickle' , 'rb'))
classes = pickle.load(open('classes.pickle' , 'rb'))
model  = load_model('chatbot_model.model')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    print(sentence_words)
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    #print(words)
    for w in  sentence_words:
        for i ,word in enumerate(words):
            if word  == w:
                bag[i] = 1
    
    return np.array(bag)


def predict_class(sentance):
    bow = bag_of_words(sentance)
    res = model.predict(np.array([bow]))[0]
    print(res)
    results = [[i,r] for i,r in enumerate(res)]
    print(results)
    results.sort(key = lambda x: x[1], reverse=True)
    print(results)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability' : str(r[1]) })
    print(return_list)
    return return_list



def get_response(intents_list , intents_json ):

    tag = intents_list[0]['intent']
    list_of_intents = intents_json["intents"]
    result = ''
    for i in list_of_intents:

        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("bot is running")
while True:
    message = input("")
    ints = predict_class(message) 
    res = get_response(ints , intents)
    print(res)
