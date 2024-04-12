import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import random
from keras.models import load_model


words = pickle.load(open("words.pkl","rb"))
classes = pickle.load(open("classes.pkl","rb"))
model = load_model("chatbot_model.h5")
data = json.loads(open('Data.json').read())

lemmatizer = WordNetLemmatizer()

def Clean_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = Clean_sentence(sentence)
    bag = [0] * len(words)
    for word in sentence_words :
        for i, w in enumerate(words):
            if word == w : 
                bag[i]= 1 
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    pred = model.predict(np.array([bow]))[0]
    err_threshold = 0.3
    results = [[i,r] for i,r in enumerate(pred) if r> err_threshold ] 
    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent" : classes[r[0]] , "probability" : str(r[1])})
    return return_list



def get_response(intents_list, intents_json): 
    if intents_list:  # Check if intents_list is not empty
        tag = intents_list[0]['intent'] 
        list_of_intents = intents_json['intents']
        for intent in list_of_intents:
            if intent['tag'] == tag:
                result = random.choice(intent['responses'])
                return result
        else: 
            print("sorry i dont understand :( )")
    


print("GO! Bot is running!")
while True:
 message= input("")
 intent = predict_class(message)
 res = get_response(intent , data)
 print(res)