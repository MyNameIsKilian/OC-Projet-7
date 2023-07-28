from flask import Flask, request
import traceback
import pandas as pd
import pickle
import numpy as np
# import json
from sklearn.neighbors import NearestNeighbors
import git

app = Flask(__name__)
app.config["DEBUG"] = True

# MAIN_COLUMNS = ['CODE_GENDER_M', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
#                 'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working',
#                 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

MAIN_COLUMNS = ['CNT_CHILDREN','APPS_EXT_SOURCE_MEAN', 'APPS_GOODS_CREDIT_RATIO',
                'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

# data = pd.read_csv('./X_train_sample.csv', index_col=[0])
# loaded_pipeline = pickle.load(open('pipeline.sav', 'rb'))
# data_scaled = loaded_pipeline.named_steps['preprocessor'].transform(data)
# df_final = pd.DataFrame(data_scaled, columns=data.columns)

@app.route("/")
def main():

    return {
        'api':'test push from local',
        # 'data_shape': df_final.shape
    }

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        repo = git.Repo('path/to/git_repo')
        origin = repo.remotes.origin
        origin.pull()
        return 'Updated PythonAnywhere successfully', 200
    else:
        return 'Wrong event type', 400

@app.route('/prediction', methods=['POST'])
def lgbm_prediction():

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
    # loaded_pipeline = pickle.load(open('pipeline.sav', 'rb'))
    # data_scaled = loaded_pipeline.named_steps['preprocessor'].transform(data)
    # df_final = pd.DataFrame(data_scaled, columns=data.columns)
    mean_df = data[MAIN_COLUMNS].mean()
    return mean_df.to_json()


@app.route("/columns/neighbors/id/<int:customer_id>", methods=['GET'])
def colmuns_neighbors(customer_id):
    """ Return the 20 nearest neighbors main columns mean values """
    data = pd.read_csv('./X_train_sample.csv', index_col=[0])
    loaded_pipeline = pickle.load(open('pipeline.sav', 'rb'))
    data_scaled = loaded_pipeline.named_steps['preprocessor'].transform(data)
    # df_final = pd.DataFrame(data_scaled, columns=data.columns)
    neighbors = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(data_scaled)
    _, neighbors_indices = neighbors.kneighbors(data_scaled)
    neighbors_df = data[MAIN_COLUMNS].iloc[neighbors_indices[customer_id]].mean()
    return neighbors_df.to_json()
