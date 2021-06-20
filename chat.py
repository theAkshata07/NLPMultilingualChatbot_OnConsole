import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from googletrans import Translator
translator = Translator()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data,strict=False)

FILE = "data.pth"
data = torch.load(FILE)


input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Medicare"
    print("Let's chat!Your safety...Our priority😊.If you want to chat with us in English,press 1.चला बोलूया! आपली सुरक्षा...आमचे प्राधान्य😊.आपणास आमच्यासह मराठी मध्ये बोलायचे असल्यास  २ दाबा. चलो बात करते हैं!आपकी सुरक्षा...हमारी प्राथमिकता😊.अगर आप हमारे साथ अंग्रेजी में चैट करना चाहते हैं, तो ३ दबाएं|")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence !="1" and sentence != "2" and sentence !="3":
            print("Oops..You have pressed incorrect digit.Please try again:)अरेरे .. आपण चुकीचा अंक दाबला आहे. कृपया पुन्हा प्रयत्न करा :)ओह..आपने गलत अंक दबाया है। कृपया पुन: प्रयास करें :)")
            continue

        if sentence =="1":
            print("Medicare: Hello..How may I help you?")
            while True:
                sentence = input("You: ")
                sentence = tokenize(sentence)
                X = bag_of_words(sentence, all_words)
                X = X.reshape(1, X.shape[0])
                X = torch.from_numpy(X).to(device)

                output = model(X)
                _, predicted = torch.max(output, dim=1)

                tag = tags[predicted.item()]

                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()]
                if prob.item() > 0.75:
                    for intent in intents['intents']:
                        if tag == intent["tag"]:
                            print(f"{bot_name}: {random.choice(intent['responses'])}")
                else:
                    print(f"{bot_name}: I don't understand.Please try to ask in other way🙂")

        if sentence =="2":
            print("Medicare: नमस्कार..मी काय मदत करु शकते?")
            while True:
                sentence = input("You: ")
                translation = translator.translate(sentence, dest='en')
                sentence = tokenize(translation.text)
                X = bag_of_words(sentence, all_words)
                X = X.reshape(1, X.shape[0])
                X = torch.from_numpy(X).to(device)

                output = model(X)
                _, predicted = torch.max(output, dim=1)

                tag = tags[predicted.item()]

                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()]
                if prob.item() > 0.75:
                    for intent in intents['intents']:
                        if tag == intent["tag"]:
                            res = random.choice(intent['responses'])
                            res1 = translator.translate(res, dest='mr')
                            #for r in res1:
                            res2 = res1.text
                            print(f"{bot_name}: {res2}")
                else:
                    print(f"{bot_name}: मला समजले नाही. कृपया इतर मार्गाने विचारण्याचा प्रयत्न करा🙂")

        if sentence =="3":
            print("Medicare: नमस्कार..मैं आपकी कैसे मदद कर सकती  हूं?")
            while True:
                sentence = input("You: ")
                translation = translator.translate(sentence, dest='en')
                sentence = tokenize(translation.text)
                X = bag_of_words(sentence, all_words)
                X = X.reshape(1, X.shape[0])
                X = torch.from_numpy(X).to(device)

                output = model(X)
                _, predicted = torch.max(output, dim=1)

                tag = tags[predicted.item()]

                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()]
                if prob.item() > 0.75:
                    for intent in intents['intents']:
                        if tag == intent["tag"]:
                            res = random.choice(intent['responses'])
                            res1 = translator.translate(res, dest='hi')
                            #for r in res1:
                            res2 = res1.text
                            print(f"{bot_name}: {res2}")
                else:
                    print(f"{bot_name}: मुझे समझ नहीं आया।कृपया दूसरे तरीके से पूछने की कोशिश करें🙂")

'''from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)
if __name__ == "__main__":
    app.run()'''

