import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Charger le fichier intents.json avec encodage correct
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)


# Préparation des données
X = []
y = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(intent["tag"])

# ✅ Utiliser CountVectorizer sans nltk
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

# Entraînement
model = MultinomialNB()
model.fit(X_vect, y)

# Fonction de réponse
def get_response(user_input):
    user_vect = vectorizer.transform([user_input])
    predicted_tag = model.predict(user_vect)[0]
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

# Lancer la conversation
print("Chatbot: Hello! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Chatbot: Bye! 👋")
        break
    response = get_response(user_input)
    print("Chatbot:", response)
