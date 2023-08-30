#%%
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

# %%
n_iter = (5, 10, 15)
lr = (0.1, 0.01)
n_layers = (2, 3, 4)
solver = ("adam", "sgd", "lbfgs")

args = list(product(n_iter, lr, n_layers, solver))

#%%
train_images_norm = StandardScaler().fit_transform(train_images.T).T
test_images_norm = StandardScaler().fit_transform(test_images.T).T

#%%
results = []
for n_iter_, lr_, n_layers_, solver_ in tqdm(args):
    clf = MLPClassifier(
        solver=solver_, 
        learning_rate_init=lr_, 
        hidden_layer_sizes=(100, n_layers_),
        max_iter=n_iter_).fit(train_images_norm, train_labels)
    #cm = confusion_matrix(test_labels, clf.predict(test_images_norm))
    #print(cm)
    report = pd.DataFrame(classification_report(test_labels, clf.predict(test_images_norm), output_dict=True))
    print(report)
    results.append(report)
    
#%%
names = ["n_iter", "lr", "n_layers", "solver", "score"]
df_report = pd.concat(results).set_index(pd.MultiIndex.from_tuples(list(product(n_iter, lr, n_layers, solver, results[0].index)), names=names)).reset_index()
df_report.to_csv("mlp_scores.csv")

#%%
names = ["n_iter", "lr", "n_layers", "solver", "score"]
df_report = pd.read_csv("mlp_scores.csv")
df_report.pivot(index=names[:-1], columns=names[-1])["weighted avg"]
