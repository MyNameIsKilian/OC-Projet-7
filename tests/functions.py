import pickle
import pandas as pd

def add_numbers(a,b):
    return a+b

def display_model_leaves():
    loaded_model = pickle.load(open('../models/lgbm_model.sav', 'rb'))
    return loaded_model.num_leaves

def predict_class():
    sample_data = pd.read_csv('../data/X_train_sample.csv', index_col=[0])
    loaded_pipeline = pickle.load(open('../models/pipeline.sav', 'rb'))
    return loaded_pipeline.predict(sample_data.iloc[1:2,:])

def display_explainer_expected_values():
    loaded_explainer = pickle.load(open('../models/shap_explainer.sav', 'rb'))
    return loaded_explainer.expected_value