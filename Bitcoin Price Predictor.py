import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
bitcoin_data = pd.read_csv("Bitcoin Historical Dataset.csv")
print("---")

# Split the data into training and testing sets
X = bitcoin_data[['Open', 'Low', 'High']]
y = bitcoin_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("---")
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("---")
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Accuracy: {:.3f}%".format(model.score(X_test,y_test)*100))

print("---")

from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html', template_folder='template')

@app.route('/predict', methods=['POST'])
def predict():
    open_price = float(request.form['open'])
    low_price = float(request.form['low'])
    high_price = float(request.form['high'])
    prediction = model.predict([[open_price, low_price, high_price]])
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
