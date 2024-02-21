from datasets.load_cifar10_samples import load_cifar10_samples
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

def extract_sift_features(img):
    sift = cv.SIFT_create()
    keypoints = sift.detect(img, None)
    return keypoints

def build_bovw_histogram(kmeans, descriptors):
    histogram = np.zeros(len(kmeans.cluster_centers_))
    labels = kmeans.predict(descriptors)
    for label in labels:
        histogram[label] += 1
    return histogram

def reshape(data):
    return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


if __name__ == "__main__":

    train_data, train_labels, test_data, test_labels = load_cifar10_samples()

    train_data = reshape(train_data)
    img = cv.cvtColor(train_data[3], 0)

    plt.imshow(img)
    plt.show()

    kp = extract_sift_features(img)

    img = cv.drawKeypoints(img, kp, img)
    cv.imwrite('sift_keypoints.jpg', img)
