import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

is_init = False
label = []
dictionary = {}
c = 0

# Load data from .npy files
for file in os.listdir():
    if file.endswith('.npy'):
        data = np.load(file)
        current_size = data.shape[0]
        current_label = file.split('.')[0]

        if not is_init:
            is_init = True
            X = data
            Y = np.array([current_label] * current_size).reshape(-1, 1)
        else:
            X = np.concatenate((X, data))
            Y = np.concatenate((Y, np.array([current_label] * current_size).reshape(-1, 1)))

        label.append(current_label)
        dictionary[current_label] = c
        c += 1


for i in range(Y.shape[0]):
    Y[i, 0] = dictionary[Y[i, 0]]

Y = np.array(Y, dtype="int32")
Y = to_categorical(Y)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X_shuffled = X[indices]
Y_shuffled = Y[indices]


ip = Input(shape=(X.shape[1],))
m = Dense(512, activation='relu')(ip)
m = Dense(256, activation='relu')(m)
op = Dense(Y.shape[1], activation='softmax')(m)  # Corrected output layer

model = Model(inputs=ip, outputs=op)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_shuffled, Y_shuffled, epochs=50)  # Use shuffled data

# Save the model and labels
model.save('model.h5')
np.save("labels.npy", np.array(label))