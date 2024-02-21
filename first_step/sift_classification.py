from datasets.load_cifar10_samples import load_cifar10_samples
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

label2string = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def extract_sift_features(img):
    sift = cv.SIFT_create()
    keypoints, desc = sift.detectAndCompute(img, None)
    return desc

def cluster_descriptors(descriptors, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(descriptors)
    return kmeans

def build_bovw_histogram(kmeans, descriptors):
    histogram = np.zeros(len(kmeans.cluster_centers_))
    labels = kmeans.predict(descriptors)
    for label in labels:
        histogram[label] += 1
    return histogram

def reshape(data):
    return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

def train_classifier(training_data, kmeans):
    X_train = []
    y_train = []

    for img, label in training_data:
        descriptors = extract_sift_features(img)
        if descriptors is not None:
            histogram = build_bovw_histogram(kmeans, descriptors)
            X_train.append(histogram)
            y_train.append(label)

    clf = KNeighborsClassifier(n_neighbors=25)# .SVC()
    clf.fit(X_train, y_train)
    return clf
def test_classifier(test_data, clf, kmeans):
    X_test = []
    y_true = []

    for img, label in test_data:
        descriptors = extract_sift_features(img)
        if descriptors is not None:
            histogram = build_bovw_histogram(kmeans, descriptors)
            X_test.append(histogram)
            y_true.append(label)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
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

    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":

    train_data, train_labels, test_data, test_labels = load_cifar10_samples()

    train_data = reshape(train_data)
    test_data = reshape(test_data)

    #descriptors
    descriptors = np.load('descriptors.npy')
    #for img in train_data:
    #    desc = extract_sift_features(img)
    #    if desc is not None:  # Check if extract_sift_features() returns a valid descriptor
    #        descriptors.extend(desc)

    #np.save('descriptors.npy', np.array(descriptors))


    k = 100  #Number of clusters
    kmeans = cluster_descriptors(np.array(descriptors), k)

    clf = train_classifier(zip(train_data, train_labels), kmeans)

    # testing
    test_classifier(zip(test_data, test_labels), clf, kmeans)

    print('done')

    #kp = extract_sift_features(img)
    #img = cv.drawKeypoints(img, kp, img)
    #cv.imwrite('sift_keypoints.jpg', img)
