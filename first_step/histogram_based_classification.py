# load cifar-10 histogram data and train a svm classifier
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

label2string = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__':
    # load the histograms
    train_hist = np.load('train_hist.npy')
    train_labels = np.load('train_labels.npy')
    test_hist = np.load('test_hist.npy')
    test_labels = np.load('test_labels.npy')

    # train a svm classifier
    print('Training SVM...')
    # time the training
    start = time.time()
    clf = LinearSVC()
    clf.fit(train_hist, train_labels)
    # clf.fit(train_hist[0:10_000], train_labels[0:10_000])
    end = time.time()
    print('SVM trained')
    print(f'Training time: {end - start:.2f} seconds')
    # predict the test data
    pred = clf.predict(test_hist)

    # calculate the accuracy
    acc = accuracy_score(test_labels, pred)
    print(f'Accuracy: {acc:.2f}')
    # print confusion matrix

    cm = confusion_matrix(test_labels, pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # add legend to the plot
    plt.legend(label2string, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

