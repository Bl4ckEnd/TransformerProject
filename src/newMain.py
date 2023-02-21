from src.newModel import make_model
from src.imports import *
from src.utilities import *


def inference_test():
    test_model = make_model(11, N=2, d_model=32, d_ff=64)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    ys = test_model(src)

    print("Output: ", ys)


if __name__ == "__main__":
    inference_test()
