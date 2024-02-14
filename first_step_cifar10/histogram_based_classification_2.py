# load cifar-10 histogram data and train a svm classifier
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

label2string = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__':
    test_hist = np.load('test_hist.npy')
    test_labels = np.load('test_labels.npy')
    # check if a model is already trained
    # if not os.path.exists(os.path.join(os.getcwd(), 'svm_model.joblib')):
    if True:
        # load the training histograms
        train_hist = np.load('train_hist.npy')
        train_labels = np.load('train_labels.npy')
        # apply PCA to the histograms
        print('Applying PCA...')
        start = time.time()
        pca = PCA(n_components=100)
        train_hist = pca.fit_transform(train_hist)
        test_hist = pca.transform(test_hist)
        print('PCA applied')
        end = time.time()
        print(f'PCA time: {end - start:.2f} seconds')

        # train a svm classifier
        print('Training SVM...')
        # time the training
        start = time.time()
        clf = LinearSVC(random_state=42, max_iter=10_000, verbose=1)
        clf.fit(train_hist, train_labels)
        # clf.fit(train_hist[0:10_000], train_labels[0:10_000])
        end = time.time()
        print('SVM trained')
        print(f'Training time: {end - start:.2f} seconds')

        # save the model
        from joblib import dump
        dump(clf, 'svm_model.joblib')
    else:
        print('Loading SVM model...')
        from joblib import load
        clf = load('svm_model.joblib')
        print('SVM model loaded')

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
    # save the confusion matrix
    plt.savefig('confusion_matrix.png')

