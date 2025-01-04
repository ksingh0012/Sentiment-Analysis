from flask import Flask, request, render_template, jsonify
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the trained model
with open('Logistic_Regression_best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|[^a-zA-Z\s]', '', text)         # Remove mentions and special characters
    text = text.lower()                                  # Convert to lowercase
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Check if cleaned_text is empty
    if not cleaned_text:
        return jsonify({'error': 'Input text is empty after cleaning.'}), 400
    
    try:
        # Transform text using TF-IDF
        transformed_text = vectorizer.transform([cleaned_text])  
        
        # Predict sentiment
        prediction = model.predict(transformed_text)
        
        # Convert prediction to sentiment label
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
        return jsonify({'prediction': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
