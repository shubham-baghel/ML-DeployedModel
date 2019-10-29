import numpy as np 
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# model = pickle.load(open('book.pkl', 'rb'))

@app.route('/', methods=['GET'])
def hello():
    return "Welcome to Book Recommendation App."
    
@app.route('/r/<int:id>/', methods=['GET'])
def predict(id):
    print(id)
    obj = recommendedBooks(model,id)
    return jsonify(obj)

@app.route('/pred/<int:id>/', methods=['GET'])
def testpredict(id):
    x = str(id) + " test"
    return jsonify(x)


def recommendedBooks(model_knn, bookIndex, number=6):
        distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[bookIndex,:].values.reshape(1, -1), n_neighbors = number)
        recommendedlist = []
        for i in range(0, len(distances.flatten())):
            if i == 0:
                print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[bookIndex]))
            else:
                print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
                recommendedlist.append(us_canada_user_rating_pivot.index[indices.flatten()[i]])

        return recommendedlist


if __name__ == '__main__':
    with open('book.pkl','rb') as f:
        model = pickle.load(f)
    with open('bookData.pkl','rb') as d:
        data = pickle.load(d)
        us_canada_user_rating_pivot = data.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
    app.run()