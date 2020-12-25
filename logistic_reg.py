import numpy as np
from sklearn.linear_model import LogisticRegression
#from sklearn.decomposition import PCA
#from sklearn.cross_decomposition import CCA
from sklearn.ensemble import AdaBoostClassifier

val = 0
train = 0
accuracies = [0] * 6

data = np.load("./data/data.npy")
labels = np.load("./data/labels.npy")
file_name = np.load("./data/filename.npy", allow_pickle=True)

N = 1
for i in range(N):
    perm = np.random.permutation(len(data))

    train_X = data[perm[len(perm) // 5:]]
    train_Y = labels[perm[len(perm) // 5:]]
    train_file = file_name[perm[len(perm) // 5:]]
    val_X = data[perm[:len(perm) // 5]]
    val_Y = labels[perm[:len(perm) // 5]]
    val_file = file_name[perm[:len(perm) // 5]]

    #reg = AdaBoostClassifier(n_estimators=100)
    #reg.fit(train_X, train_Y)

    reg = LogisticRegression(max_iter = 500).fit(train_X, train_Y)

    pred_Y = reg.predict(val_X)
    val += np.sum(pred_Y == val_Y)/len(val_Y)
    for i in range(6):
        accuracies[i] += np.sum(pred_Y[val_Y == i] == i) / np.sum(val_Y==i)

    """val_X, val_Y = train_X, train_Y
    pred_Y = reg.predict(val_X)
    train += np.sum(pred_Y == val_Y)/len(val_Y)"""

#print(val / N, train / N)
print(np.array(accuracies) / N)
print(val_file[val_Y != pred_Y])
