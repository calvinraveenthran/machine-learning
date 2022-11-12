import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([
    [200.0, 17.0],
    [120.0, 5.0],
    [425.0, 20.0],
    [212.0, 18.0],
])
y = np.array([1,0,0,1])

model = Sequential([
    Dense(units=3, activation='sigmoid'),
    Dense(units=1, activation='sigmoid')
])

model.compile(...)
model.fit(x,y)
