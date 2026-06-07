from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive MRI image
    # Run AI model
    # Return result
    return render_template('result.html')

app.run(debug=True)