# from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
import tensorflow as tf

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

import os, re
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, Reshape, InputLayer
from keras.optimizers import SGD

def data():
    import os, re
    import numpy as np
    import pandas as pd
    from scipy.misc import imread
    from sklearn.metrics import accuracy_score
    from sklearn.cross_validation import train_test_split

    root = "/home/ubuntu/music/Checked/"
    path = os.path.join(root, "targetdirectory")
    list1 = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith(".png"):
                cla = re.search('Checked/(\d+)', os.path.join(path))
                path1 = os.path.join(path) + '/' + name
                list1.append([os.path.join(name), path1, cla.group(1)])
    dff = pd.DataFrame(list1, columns=['name','path','key'])

    temp = []
    i = 0
    for img_name in dff['path']:
        if os.path.isfile(img_name):
            image_path = os.path.join(img_name)
            img = imread(image_path, flatten=True)
            img = img.astype('float32')
            temp.append(img)
            i += 1


    y = dff['key'].values
    x=np.stack(temp)
    x = x.reshape(i, 300, 300, 1).astype('Float32')
    x /= 255.0
    x = 1.0 - x
    xtr, xte, ytr, yte = train_test_split(x, y, stratify=y)
    ytr = keras.utils.np_utils.to_categorical(ytr, 12)
    yte = keras.utils.np_utils.to_categorical(yte, 12)
    return xtr, ytr, xte, yte


def model(xtr, ytr, xte, yte):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Convolution2D(12, 4, 3, input_shape=(300, 300, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense({{choice([120, 130])}}))
    model.add(Activation('relu'))
    model.add(Dense(12))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer='rmsprop')

    model.fit(xtr, ytr,
              batch_size={{choice([50, 64])}},
              epochs=10,
              verbose=2,
              validation_data=(xte, yte))
    score, acc = model.evaluate(xte, yte, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}




if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    xtr, ytr, xte, yte = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(xte, yte))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
