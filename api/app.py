from flask import Flask, request
import traceback
import pandas as pd
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import subprocess
import git

app = Flask(__name__)

MAIN_COLUMNS = ['CNT_CHILDREN','APPS_EXT_SOURCE_MEAN', 'APPS_GOODS_CREDIT_RATIO',
                'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

@app.route("/")
def main():
    """ Return simple response to test APÏ """
    return {
        'api':'push from local to github, need to pull it from pythonanywhere',
    }
    
@app.route('/pull', methods=['GET'])
def git_pull():
    """ Pull GitHub code """
    project_path = "../OC-Projet-7/"
    remote_name = "origin"
    try:
        subprocess.run(['git', 'rev-parse'], cwd=project_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(['git', 'pull', remote_name], cwd=project_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("Git pull réussi.")
        return 'Updated PythonAnywhere successfully', 200
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors du git pull : {e}")
        return 'Error during pull command', 400
    except Exception as ex:
        print(f"Erreur inattendue : {ex}")
        return 'Error unknown', 400

@app.route('/prediction', methods=['POST'])
def lgbm_prediction():
    """ Return probabilties to predict class 0 and 1 """
    try:
        row_data = request.get_json()
        loaded_pipeline = pickle.load(open('pipeline.sav', 'rb'))
        row_df = pd.DataFrame([row_data], index=[0])
        prediction = loaded_pipeline.predict_proba(row_df)

        response = {
            'status_code': 200,
            'body': prediction.tolist()
        }
        return response

    except Exception as e:
        traceback.print_exc()

        response = {
            'status_code': 500,
            'body': str(e)
        }
        return response

@app.route('/shap-values', methods=['POST'])
def load_shap_values():
    """ Return shap values of the customer """
    try:
        row_data = request.get_json()
        row_df = pd.DataFrame([row_data], index=[0])
        loaded_pipeline = pickle.load(open('pipeline.sav', 'rb'))
        explainer = pickle.load(open('shap_explainer.sav', 'rb'))
        row_scaled = loaded_pipeline.named_steps['preprocessor'].transform(row_df)

        shap_values = explainer.shap_values(row_scaled)
        expected_value = explainer.expected_value

        response = {
            'status_code': 200,
            'body': {
                'shap_values': np.array(shap_values).tolist(),
                'expected_value': expected_value,
            }
        }
        return response

    except Exception as e:
        traceback.print_exc()

        response = {
            'status_code': 500,
            'body': str(e)
        }
        return response

@app.route("/columns/mean", methods=['GET'])
def colmuns_mean():
    """ Return the main columns mean values """
    data = pd.read_csv('./X_train_sample.csv', index_col=[0])
    mean_df = data[MAIN_COLUMNS].mean()
    return mean_df.to_json()

@app.route("/columns/neighbors/id/<int:customer_id>", methods=['GET'])
def colmuns_neighbors(customer_id):
    """ Return the 20 nearest neighbors main columns mean values """
    data = pd.read_csv('./X_train_sample.csv', index_col=[0])
    loaded_pipeline = pickle.load(open('pipeline.sav', 'rb'))
    data_scaled = loaded_pipeline.named_steps['preprocessor'].transform(data)
    neighbors = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(data_scaled)
    _, neighbors_indices = neighbors.kneighbors(data_scaled)
    neighbors_df = data[MAIN_COLUMNS].iloc[neighbors_indices[customer_id]].mean()
    return neighbors_df.to_json()
