# load cifar-10 histogram data and train a svm classifier
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score
# import knn classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

label2string = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__':
    test_hist = np.load('test_hist.npy')
    test_labels = np.load('test_labels.npy')
    # check if a model is already trained
    if not os.path.exists(os.path.join(os.getcwd(), 'knn_model.joblib')):
        # load the training histograms
        train_hist = np.load('train_hist.npy')
        train_labels = np.load('train_labels.npy')

        # create a classifier
        print('Creating KNN...')
        clf = KNeighborsClassifier(n_neighbors=25)
        clf.fit(train_hist, train_labels)

        # save the model
        from joblib import dump
        dump(clf, 'knn_model.joblib')
    else:
        print('Loading KNN model...')
        from joblib import load
        clf = load('KNN_model.joblib')
        print('KNN model loaded')

    # predict the test data
    print('Predicting test data...')
    start = time.time()
    pred = clf.predict(test_hist)
    end = time.time()
    print(f'Prediction time: {end - start:.2f} seconds')

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
    # save the confusion matrix
    plt.savefig('confusion_matrix_knn.png')

