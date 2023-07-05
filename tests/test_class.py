from functions import *

class TestClass:
    def test_add_numbers(self):
        assert add_numbers(2, 3) == 5

    def test_model_leaves(self):
        assert display_model_leaves() == 30

    def test_predict_class(self):
        assert predict_class() == [0]