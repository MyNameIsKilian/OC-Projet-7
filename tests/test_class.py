from functions import *
from IPython.display import display

class TestClass:
    def test_add_numbers(self):
        assert add_numbers(2, 3) == 5

    def test_model_leaves(self):
        assert display_model_leaves() == 30

    def test_predict_class(self):
        assert predict_class() == [0]

    def test_expected_value_from_explainer(self):
        assert display_explainer_expected_values() == [0.36253430662611813, -0.36253430662611813]
