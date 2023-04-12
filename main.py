# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:08:36 2023

@author: prash
"""

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
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase import firebase

firebase = firebase.FirebaseApplication('https://vocals-e4589-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words

def bag_of_words(sentence,words):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence,model,words,classes):
    bow = bag_of_words(sentence,words)
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
def firebase_response(entry):
    result = firebase.get('/prash/-NPmLDGT5Mvob4GFXAe_', None)
    res1 = result[entry]
    engine = pyttsx3.init()
    engine.say(res1)
    return res1
    
def chatbot_response(msg,model,intents,words,classes):
    if msg.lower() == "bye" or msg.lower()=="goodbye":
        ints = predict_class(msg, model,words,classes)
        res = "bye"
        return res
    
    else:
        ints = predict_class(msg, model,words,classes)
        res = get_response(ints, intents)
        if res == "name to be accessed":
            entry = 'name'
            return firebase_response(entry)
        elif res == "date to be accessed":
            entry = 'date'
            return firebase_response(entry)
        elif res == "time to be accessed":
            entry = 'time'
            return firebase_response(entry)
        elif res == "address to be accessed":
            entry = 'address'
            return firebase_response(entry)
        elif res == "number to be accessed":
            entry = 'person_no'
            return firebase_response(entry)
        elif res == "dept to be accessed":
            entry = 'department'
            return firebase_response(entry)
        elif res == "docname to be accessed":
            entry = 'doctor_name'
            return firebase_response(entry)
        elif res == "members to be accessed":
            entry = 'no_people'
            return firebase_response(entry)
        elif res == "issue to be accessed":
            entry = 'problem'
            return firebase_response(entry)
        elif res == "type to be accessed":
            entry = ''
            return firebase_response(entry)
        else:
            engine = pyttsx3.init()
            engine.say(res)
            engine.runAndWait()
            return res
        
''' Flask code '''

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return 'Book Your Appointments'

#function to replace '+' character with ' ' spaces
def decrypt(msg):
    
    string = msg
    
    #converting back '+' character back into ' ' spaces
    #new_string is the normal message with spaces that was sent by the user
    new_string = string.replace("+", " ")
    
    return "hello " 

#here we will send a string from the client and the server will return another
#string with som modification
#creating a url dynamically
@app.route('/electrician/<name>', methods= ['GET']) 
def electrician(name):
    
    
    #dec_msg is the real question asked by the user
    dec_msg = name.replace("+", " ")
    
    intents = json.loads(open("E:\Major Project\AI chatbot\electrician.json").read())
    words = pickle.load(open('electrician_words.pkl', 'rb'))
    classes = pickle.load(open('electrician_classes.pkl', 'rb'))
    model = load_model('electrician_chatbotmodel.h5')
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg,model,intents,words,classes)
    
    # retrieve the user's data from the Realtime Database
    #ref = db.reference('/prash/-NPmLDGT5Mvob4GFXAe_',None)
    #user_data = ref.get()
    result = firebase.get('/prash/-NPmLDGT5Mvob4GFXAe_', None)
    print(str(result))
    #name = user_data['name']
    #time = user_data['time']
    
    #print('Name:', name)
    #print('Time:', time)
    #print(str(user_data))
    #print(user_data)
    
    
    if response == "bye":
        # exit the application if the user says "bye"
        request.environ.get('werkzeug.server.shutdown')()
    
    
    return response

@app.route('/vehicle/<name>', methods= ['GET']) 
def vehicle(name):
    
    #dec_msg is the real question asked by the user
    dec_msg = name.replace("+", " ")
    
    intents = json.loads(open("E:\Major Project\AI chatbot\vehicle.json").read())
    words = pickle.load(open('vehicle_words.pkl', 'rb'))
    classes = pickle.load(open('vehicle_classes.pkl', 'rb'))
    model = load_model('vehicle_chatbotmodel.h5')
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg,model,intents,words,classes)
    
    if response == "bye":
        # exit the application if the user says "bye"
        request.environ.get('werkzeug.server.shutdown')()
    
    
    return response

@app.route('/hospital/<name>', methods= ['GET']) 
def hospital(name):
    
    #dec_msg is the real question asked by the user
    dec_msg = name.replace("+", " ")
    
    intents = json.loads(open("E:\Major Project\AI chatbot\hospital.json").read())
    words = pickle.load(open('hospital_words.pkl', 'rb'))
    classes = pickle.load(open('hospital_classes.pkl', 'rb'))
    model = load_model('hospital_chatbotmodel.h5')
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg,model,intents,words,classes)
    
    if response == "bye":
        # exit the application if the user says "bye"
        request.environ.get('werkzeug.server.shutdown')()
    
    
    return response

@app.route('/restaurant/<name>', methods= ['GET']) 
def restaurant(name):
    
    #dec_msg is the real question asked by the user
    dec_msg = name.replace("+", " ")
    
    intents = json.loads(open("E:\Major Project\AI chatbot\restaurant.json").read())
    words = pickle.load(open('restaurant_words.pkl', 'rb'))
    classes = pickle.load(open('restaurant_classes.pkl', 'rb'))
    model = load_model('restaurant_chatbotmodel.h5')
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg,model,intents,words,classes)
    
    if response == "bye":
        # exit the application if the user says "bye"
        request.environ.get('werkzeug.server.shutdown')()
    
    
    return response

@app.route('/salon/<name>', methods= ['GET']) 
def salon(name):
    
    #dec_msg is the real question asked by the user
    dec_msg = name.replace("+", " ")
    
    intents = json.loads(open("E:\Major Project\AI chatbot\salon.json").read())
    words = pickle.load(open('salon_words.pkl', 'rb'))
    classes = pickle.load(open('salon_classes.pkl', 'rb'))
    model = load_model('salon_chatbotmodel.h5')
    
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg,model,intents,words,classes)
    
    if response == "bye":
        # exit the application if the user says "bye"
        request.environ.get('werkzeug.server.shutdown')()
    
    
    return response

@app.route('/dentist/<name>', methods= ['GET']) 
def dentist(name):
    
    #dec_msg is the real question asked by the user
    dec_msg = name.replace("+", " ")
    
    intents = json.loads(open("E:\Major Project\AI chatbot\dentist.json").read())
    words = pickle.load(open('dentist_words.pkl', 'rb'))
    classes = pickle.load(open('dentist_classes.pkl', 'rb'))
    model = load_model('dentist_chatbotmodel.h5')
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg,model,intents,words,classes)
    
    if response == "bye":
        # exit the application if the user says "bye"
        request.environ.get('werkzeug.server.shutdown')()
    
    
    return response

@app.route('/plumber/<name>', methods= ['GET']) 
def plumber(name):
    
    #dec_msg is the real question asked by the user
    dec_msg = name.replace("+", " ")
    
    intents = json.loads(open("E:\Major Project\AI chatbot\plumber.json").read())
    words = pickle.load(open('plumber_words.pkl', 'rb'))
    classes = pickle.load(open('plumber_classes.pkl', 'rb'))
    model = load_model('plumber_chatbotmodel.h5')
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg,model,intents,words,classes)
    
    if response == "bye":
        # exit the application if the user says "bye"
        request.environ.get('werkzeug.server.shutdown')()
       
    return response

@app.route('/emergency/<name>', methods= ['GET']) 
def emergency(name):
    
    #dec_msg is the real question asked by the user
    dec_msg = name.replace("+", " ")
    
    intents = json.loads(open("E:\Major Project\AI chatbot\emergency.json").read())
    words = pickle.load(open('emergency_words.pkl', 'rb'))
    classes = pickle.load(open('emergency_classes.pkl', 'rb'))
    model = load_model('emergency_chatbotmodel.h5')
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg,model,intents,words,classes)
    
    if response == "bye":
        # exit the application if the user says "bye"
        request.environ.get('werkzeug.server.shutdown')()   
    
    return response

if __name__ == '__main__':
    app.run(debug=True, threaded = False)
