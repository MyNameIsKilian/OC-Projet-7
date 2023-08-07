import pickle
import pandas as pd

loaded_model = pickle.load(open('/home/kilian/workspaces/workspace-data/OC-P7/models/lgbm_model.sav', 'rb'))

def add_numbers(a,b):
    return a+b

def display_model_leaves():
    return loaded_model.num_leaves

def predict_class():
    sample_data = pd.read_csv('/home/kilian/workspaces/workspace-data/OC-P7/data/X_train_sample.csv', index_col=[0])
    loaded_pipeline = pickle.load(open('/home/kilian/workspaces/workspace-data/OC-P7/models/pipeline.sav', 'rb'))
    return loaded_pipeline.predict(sample_data.iloc[1:2,:])

def display_explainer_expected_values():
    loaded_explainer = pickle.load(open('/home/kilian/workspaces/workspace-data/OC-P7/models/shap_explainer.sav', 'rb'))
    return loaded_explainer.expected_value