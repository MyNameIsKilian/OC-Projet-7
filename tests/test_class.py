from  functions import *

class TestClass:
    def test_add_numbers(self):
        print("running add_numbers function")
        assert add_numbers(2, 3) == 5

    # def test_model_leaves(self):
    #     print("running display_model_leaves function")
    #     assert display_model_leaves() == 30