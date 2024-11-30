import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# Load preprocessed data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'], dtype=int)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Build a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(42,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Save the model
model.save('gesture_model.h5')
