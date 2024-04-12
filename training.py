import random
import json
import numpy as np
import pickle
import tensorflow as tf
import nltk

from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import matplotlib.pyplot as plt



lemmatizer = WordNetLemmatizer()

intents = json.loads(open('Data.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', ',', '!', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        documents.append((word_list, intent['tag']))
        words.extend(word_list)  # Append words to the 'words' list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Create a bag of words with 1s and 0s
    bag = [1 if word in word_patterns else 0 for word in words]

    # Create the output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    # Append the bag and output_row as separate lists
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Split the training data into X and Y
data_x = np.array([item[0] for item in training])
data_y = np.array([item[1] for item in training])
train_x, test_x = train_test_split(data_x, test_size=0.1, random_state=42)
train_y, test_y = train_test_split(data_y, test_size=0.1, random_state=42)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(len(train_x[0]),), activation='relu' ))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 

hist = model.fit(train_x, train_y, epochs=200, batch_size=10 , verbose=1, validation_data=(test_x, test_y))

# Plotting accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("model accuracy.png")
plt.show()

# Plotting loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Model Loss.png')
plt.show()

plt.show()
loss , accuracy = model.evaluate(test_x, test_y)
model.save('Chatbot_model.h5', hist)

print ("the accuracy on testing data is ", accuracy)
print("Done!")
