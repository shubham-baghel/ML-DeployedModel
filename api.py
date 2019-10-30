import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

@app.before_first_request
def load_models():
    global model
    global data
    global us_canada_user_rating_pivot
    global uniqueBooks
    with open('book.pkl','rb') as f:
        model = pickle.load(f)
        
    with open('bookData.pkl','rb') as d:
        data = pickle.load(d)
        uniqueBooks =  data.drop_duplicates(subset=['ISBN','imageUrlL'])
        us_canada_user_rating_pivot = data.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html', all_books = uniqueBooks[['ISBN','imageUrlL']].to_dict('records'))
    
@app.route('/r/<string:id>/', methods=['GET'])
def predict(id):
    bookTitle = data[data["ISBN"] == id].iloc[0,:]["bookTitle"]
    distances, indices = model.kneighbors(us_canada_user_rating_pivot.loc[bookTitle,:].values.reshape(1, -1), n_neighbors = 6)
    recommendedlist = []
    selectedBook = []
    for i in range(0, len(distances.flatten())):
        name = us_canada_user_rating_pivot.index[indices.flatten()[i]]
        bookImageURL = data[data.bookTitle == name].imageUrlL.iloc[0]
        ISBN = data[data.bookTitle == name].ISBN.iloc[0]
        if i == 0:
            selectedBook.append({"name" : name, "imageUrl":bookImageURL, "ISBN": ISBN})
        else:
            recommendedlist.append({"name" : name, "imageUrl":bookImageURL, "ISBN": ISBN})
    return render_template('view.html', result = recommendedlist, selectedBook = selectedBook)

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