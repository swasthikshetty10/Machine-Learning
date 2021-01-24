from tensorflow.keras.layers import Dense, Activation, Dropout
import pickle
import json
import random
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
import nltk
# --> works, work, working , worked == work
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.python.keras.engine.sequential import relax_input_shape
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
with open("intents.json") as file:
    intents = json.load(file)

words = []
classes = []
documents = []  # documents of patterns's
ignore_letters = ['?', '.', '!', ',']
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
