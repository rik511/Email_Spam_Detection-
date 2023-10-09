from flask import Flask, render_template, request # Import Flask
import pandas as pd # Import Pandas for data manipulation using dataframes
from sklearn.feature_extraction.text import CountVectorizer # Import CountVectorizer
from nltk.tokenize import RegexpTokenizer # Import RegexpTokenizer
from nltk.stem import PorterStemmer# Import PorterStemmer
from sklearn.linear_model import PassiveAggressiveClassifier # Import PassiveAggressiveClassifier
# Create an instance of Flask
app = Flask(__name__) # Create a Flask app
# Load the dataset
df = pd.read_csv("C:\\Users\\rikha\\Documents\\data science\\email spam\\spam_ham_dataset.csv")
# Define a function to clean the text
def clean_str(string, reg=RegexpTokenizer(r'[a-z]+')): 
    string = string.lower() # Convert to lowercase
    tokens = reg.tokenize(string) # Split the string using the tokenizer
    return " ".join(tokens) # Join the tokens back into a string
# Apply the function
df['text'] = df['text'].apply(lambda string: clean_str(string)) 
# Stem the text
stemmer = PorterStemmer()
# Define a function to stem the text
def stemming(text):
    return ''.join([stemmer.stem(word) for word in text]) # Join the tokens back into a string
# Apply the function
df['text'] = df['text'].apply(stemming)
# Split the dataset into training and testing sets 
X = df['text']   
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(df['text']) # Fit the CountVectorizer to the training data
model = PassiveAggressiveClassifier(max_iter=50) # Create a PassiveAggressiveClassifier with maximum iterations of 50 
model.fit(X, y) # Fit the model to the training data
# Define a function to classify the email 
@app.route('/', methods=['GET', 'POST']) # Define a route for the application 
def index():# Define a function to classify the email
    result = None# Define a variable to store the result
    if request.method == 'POST':# Check if the request method is POST
        # Get the user input
        user_input = request.form['email_text']
        # Clean the user input
        user_input = clean_str(user_input)
        # Stem the user input 
        user_input = stemming(user_input)
        # Convert the user input to a vector
        user_input_vec = cv.transform([user_input])
        # Predict the result 
        result = model.predict(user_input_vec)[0]

    return render_template('index.html', result=result) # Render the template with the result 
# Run the app
if __name__ == '__main__':
    app.run(debug=True)