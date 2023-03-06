import random
import json
import pickle
import numpy as np
import nltk
import pyttsx3
import speech_recognition as sr
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("E:\Major Project\AI chatbot\intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence,model):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    if msg.lower() == "bye" or msg.lower()=="goodbye":
        ints = predict_class(msg, model)
        res = "bye"
        return res
    
    else:
        ints = predict_class(msg, model)
        res = get_response(ints, intents)
        engine = pyttsx3.init()
        engine.say(res)
        engine.runAndWait()
        return res


''' Flask code '''

from flask import Flask, jsonify


app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def hello():
    return jsonify({"key" : "home page value"})

#function to replace '+' character with ' ' spaces
def decrypt(msg):
    
    string = msg
    
    #converting back '+' character back into ' ' spaces
    #new_string is the normal message with spaces that was sent by the user
    new_string = string.replace("+", " ")
    
    return new_string

#here we will send a string from the client and the server will return another
#string with som modification
#creating a url dynamically
@app.route('/<name>') 
def hello_name(name):
    
    #dec_msg is the real question asked by the user
    dec_msg = decrypt(name)
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg)
    
    
    return response



if __name__ == '__main__':
    app.run(debug=True)