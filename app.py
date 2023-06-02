from flask import jsonify
import json
import werkzeug
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# app = Flask(__name__)

# @app.route("/iris", methods=["GET"])
def main():
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
    
    return jsonify(response)

    # pickle.save
    # pickle.load
    # load model from s3





