from typing import Sequence
import nltk
import json
import numpy as np
from pyjokes import jokes_de
import tensorflow
from tensorflow import keras
import pickle
import random
from tensorflow.keras.preprocessing.text import Tokenizer
sentence = """hello. buddy"""
tokens = nltk.word_tokenize(sentence)
print(tokens)


def prediction(query):
    intents = None
    with open("intents.json") as file:
        data = json.load(file)
    model = keras.models.load_model('chat_model')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    max_len = 20
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([query]),
                                                                      truncating='post', maxlen=max_len))
    # print(tensorflow.Tensor(result))
    # print(np.max(result))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    for i in data['intents']:
        if i['tag'] == tag:
            intents = [np.random.choice(i['responses']), str(tag)]
    return intents


print(prediction(sentence))
