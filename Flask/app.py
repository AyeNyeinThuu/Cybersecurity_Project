#from flask import Flask, render_template, request
#import pickle
#import os

#app = Flask(__name__)

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#model_path = os.path.join(BASE_DIR, "TF_IDF_Model.pkl")

#with open(model_path, "rb") as f:
    #model = pickle.load(f)

#@app.route("/")
#def home():
    #return "<h1>Flask is working</h1>"
    #return render_template("index.html")

#@app.route("/verify", methods=["POST"])
#def verification():
    #email_text = request.form["email_text"]

    #verification = model.predict([email_text])[0]

    #result = "Phishing" if verification == 1 else "Not Phishing"

    #return render_template("index.html", verification_text=result)

#if __name__ == "__main__":
    #app.run(debug=True)


from flask import Flask, render_template, request
import os
import pickle
import string
import nltk
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer

# Ensure required NLTK resources exist on Render
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

app = Flask(__name__)

# Load trained model using safe absolute path
model_path = os.path.join(os.path.dirname(__file__), "TF_IDF_Model.pkl")
with open(model_path, "rb") as f:
    pipeline = pickle.load(f)

# Same preprocessing used during training
stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}

def preprocess(text):
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Prediction wrapper for raw text
def predict_proba_raw(texts):
    cleaned_texts = [preprocess(t) for t in texts]
    return pipeline.predict_proba(cleaned_texts)

def predict_raw(texts):
    cleaned_texts = [preprocess(t) for t in texts]
    return pipeline.predict(cleaned_texts)

# LIME explainer
explainer = LimeTextExplainer(class_names=["Legitimate", "Phishing"])

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    lime_html = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form["email_text"]

        # Predict
        proba = predict_proba_raw([user_text])[0]
        threshold = 0.30

        if proba[1] >= threshold:
            prediction = "Phishing"
            confidence = round(proba[1] * 100, 2)
        else:
            prediction = "Legitimate"
            confidence = round(proba[0] * 100, 2)

        # Generate LIME explanation
        exp = explainer.explain_instance(
            user_text,
            predict_proba_raw,
            num_features=10
        )
        lime_html = exp.as_html()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        lime_html=lime_html,
        user_text=user_text
    )

if __name__ == "__main__":
    app.run(debug=True)