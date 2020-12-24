import numpy as np
from sklearn.linear_model import LogisticRegression
#from sklearn.decomposition import PCA
#from sklearn.cross_decomposition import CCA
from sklearn.ensemble import AdaBoostClassifier

val = 0
train = 0
for i in range(5):
    data = np.load("./data/data.npy")
    labels = np.load("./data/labels.npy")
    perm = np.random.permutation(len(data))

    train_X = data[perm[len(perm) // 5:]]
    train_Y = labels[perm[len(perm) // 5:]]
    val_X = data[perm[:len(perm) // 5]]
    val_Y = labels[perm[:len(perm) // 5]]

    reg = AdaBoostClassifier(n_estimators=100)
    reg.fit(train_X, train_Y)

    #reg = LogisticRegression(max_iter = 500).fit(train_X, train_Y)

    pred_Y = reg.predict(val_X)
    val += np.sum(pred_Y == val_Y)/len(val_Y)

    val_X, val_Y = train_X, train_Y
    pred_Y = reg.predict(val_X)
    train += np.sum(pred_Y == val_Y)/len(val_Y)

print(val / 5, train / 5)
