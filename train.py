from data_process import gen_train_test
from config import *
from model import *
import keras

X_train, X_test, y_train, y_test = gen_train_test()

model = get_model()
model.compile(loss='mean_squared_error', metrics=['mae'],
              optimizer=keras.optimizers.Adadelta())
model.summary()

model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE)
model.save(SAVE_NAME)
