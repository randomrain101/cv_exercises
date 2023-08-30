#%%
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.activations import relu
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# %%
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[1]))
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[1]))

train_images_norm = StandardScaler().fit_transform(train_images.T).T
test_images_norm = StandardScaler().fit_transform(test_images.T).T

X_train = train_images_norm.reshape(train_images_norm.shape[0], 28, 28)
X_test = test_images_norm.reshape(test_images_norm.shape[0], 28, 28)

#%%
@dataclass
class CNN:
    kernel_size: tuple = (3, 3)
    pooling_size: tuple = (2, 2)

    def fit(self, X_train, train_labels):
        with tf.device("/cpu:0"):
            model = Sequential()
            model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation=relu))
            model.add(Dropout(0.2))
            model.add(Dense(10,activation=tf.nn.softmax))

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
            model.fit(x=X_train,y=train_labels, epochs=10)
            self.model_ = model
            return self
    
    def predict(self, X_test):
        with tf.device("/cpu:0"):
            return self.model_.predict(X_test)

# %%
kernel_size = ((3, 3), (4, 4), (5, 5))
pooling_size = ((3, 3), (4, 4), (5, 5))

args = list(product(kernel_size, pooling_size))

#%%
results = []
for k_size, p_size in tqdm(args):
    print(k_size, p_size)
    clf = CNN(kernel_size=k_size, pooling_size=p_size).fit(X_train, train_labels)
    #cm = confusion_matrix(test_labels, clf.predict(test_images_norm))
    #print(cm)
    report = pd.DataFrame(classification_report(test_labels, clf.predict(X_test).argmax(axis=1), output_dict=True))
    print(report)
    results.append(report)
    
#%%
names = ["kernel_size", "pooling_size", "score"]
df_report = pd.concat(results).set_index(pd.MultiIndex.from_tuples(list(product(kernel_size, pooling_size, results[0].index)), names=names)).reset_index()
df_report.to_csv("cnn_scores.csv")

#%%
df_report.pivot(index=names[:-1], columns=names[-1])["weighted avg"]

#%%
clf_best = CNN((5, 5), (3, 3)).fit(X_train, train_labels)

# %%
X_missclassified = X_test[clf_best.predict(X_test).argmax(axis=1) != test_labels]

# %%
for pic in X_missclassified:
    plt.imshow(pic)
    plt.show()
