from data_process import *
from config import *
from model import *
import keras
import numpy as np


model = keras.models.load_model(SAVE_NAME)
_, X_test, _, y_test = gen_train_test()
print(model.evaluate(X_test, y_test, verbose=VERBOSE))


