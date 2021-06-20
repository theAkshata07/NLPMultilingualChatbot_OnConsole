import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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
print("Let's chat!Your safety...Our priority😊Type quit to exit.If you want to chat with us in English,press 1.चला बोलूया! आपली सुरक्षा...आमचे प्राधान्य😊बाहेर पडण्यासाठी बाहेर पडा असे टाइप करा.आपणास आमच्यासह मराठी मध्ये बोलायचे असल्यास  २ दाबा")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit" or "बाहेर पडा":
        break
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