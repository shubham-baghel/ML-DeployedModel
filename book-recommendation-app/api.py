import numpy as np 
from flask import Flask, request, jsonify
import pickle
import model


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return "Welcome to Book Recommendation App."
    
@app.route('/r/<int:id>/', methods=['GET'])
def predict(id):
    print(id)
    obj = model()
    res = obj.recommendedBooks(id)
    return jsonify(res)

if __name__ == '__main__':
    with open('book.pkl','rb') as f:
        model = pickle.load(f)
    app.run(port=5000, debug=True)