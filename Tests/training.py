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


for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # tokenizes the  word for ez  how's going? --> "how" "s" "going" "?"
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# print(documents)
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
# print(words)
word = sorted(set(words))
classes = sorted(set(classes))
print(classes)
pickle.dump(words, open('words.pickle', 'wb'))
pickle.dump(classes, open('classes.pickle', 'wb'))


training = []
output_empty = [0]*len(classes)
print(output_empty)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word[0].lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index((document[1]))] = 1
    training.append([bag, output_row])
print()
random.shuffle(training)

training = np.array(training)
# print(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
# print(train_x)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=3000, batch_size=100, verbose=1)
model.save('chatbot_model.model', hist)
print('done')
