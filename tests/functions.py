import pickle

loaded_model = pickle.load(open('lgbm_model.sav', 'rb'))
# loaded_pipeline = pickle.load(open('pipeline.sav', 'rb'))
#
def add_numbers(a,b):
    return a+b

def display_model():
    print(loaded_model)
    return True

def main():
    display_model()