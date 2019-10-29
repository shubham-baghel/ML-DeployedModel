import numpy as np 
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

@app.before_first_request
def load_models():
    global model
    global us_canada_user_rating_pivot
    with open('book.pkl','rb') as f:
        model = pickle.load(f)
        
    with open('bookData.pkl','rb') as d:
        data = pickle.load(d)
        us_canada_user_rating_pivot = data.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)

@app.route('/', methods=['GET'])
def hello():
    return "Welcome to Book Recommendation App."
    
@app.route('/r/<int:id>/', methods=['GET'])
def predict(id):
   
    distances, indices = model.kneighbors(us_canada_user_rating_pivot.iloc[id,:].values.reshape(1, -1), n_neighbors = 6)
    recommendedlist = []
    for i in range(0, len(distances.flatten())):
        if i != 0:
            recommendedlist.append(us_canada_user_rating_pivot.index[indices.flatten()[i]])
    return jsonify(recommendedlist)

@app.route('/pred/<int:id>/', methods=['GET'])
def testpredict(id):
    x = str(id) + " test"
    return jsonify(x)


def recommendedBooks(bookIndex, number=6):
        distances, indices = model.kneighbors(us_canada_user_rating_pivot.iloc[bookIndex,:].values.reshape(1, -1), n_neighbors = number)
        recommendedlist = []
        for i in range(0, len(distances.flatten())):
            if i != 0:
                recommendedlist.append(us_canada_user_rating_pivot.index[indices.flatten()[i]])
        return recommendedlist


if __name__ == '__main__':
    app.run(debug=True)