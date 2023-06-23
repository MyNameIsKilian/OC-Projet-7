from flask import Flask
import json
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route("/")
def main():

    return {
        'api':'return from GitHub pull 23/06 12:24'
    }

@app.route('/iris')
def iris_prediction():

    resp = load_iris()
    return resp

@app.route('/data')
def iris_prediction():

    resp = load_my_dataframe()
    return resp


def load_iris():
    # Charger le dataset Iris
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_params = model.get_params()
    json_params = json.dumps(model_params)

    response = {
        'dataset': 'iris',
        'model_parameters': json_params,
        'accuracy': accuracy
    }

    return response

def load_my_dataframe():
    # return jsonify({'prediction': prediction.tolist()})
    data = pd.read_csv('../X_train_dataframe.csv')
    loaded_pipeline = pickle.load(open('pipeline.sav', 'rb'))
    return {'first_value': data.iloc[0:1,:].values}