# create the histograms for the cifar10 dataset

from datasets.load_cifar10_samples import load_cifar10_samples
import numpy as np
import matplotlib.pyplot as plt
import os

# create rgb histogram
def create_rgb_histograms(data):
    histograms = []
    for img in data:
        red_hist = np.histogram(img[0:1024], bins=256, range=[0, 256])[0]
        green_hist = np.histogram(img[1024:2048], bins=256, range=[0, 256])[0]
        blue_hist = np.histogram(img[2048:], bins=256, range=[0, 256])[0]
        histograms.append(np.concatenate((red_hist, green_hist, blue_hist)))
    return np.array(histograms)

def label2string(label):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return labels[label]

if __name__ == '__main__':
    # check if the histograms are already created and load them
    if not os.path.exists(os.path.join(os.getcwd(), 'train_hist.npy')):
        print('Creating histograms...')
        # load the cifar10 dataset
        train_data, train_labels, test_data, test_labels = load_cifar10_samples()
        # create the histograms
        train_hist = create_rgb_histograms(train_data)
        test_hist = create_rgb_histograms(test_data)

        # save the histograms
        np.save('train_hist.npy', train_hist)
        np.save('train_labels.npy', train_labels)
        np.save('test_hist.npy', test_hist)
        np.save('test_labels.npy', test_labels)
    else:
        train_hist = np.load('train_hist.npy')
        train_labels = np.load('train_labels.npy')
        test_hist = np.load('test_hist.npy')
        test_labels = np.load('test_labels.npy')
        print('Histograms already created')
        print('Train data shape:', train_hist.shape)
        print('Train labels shape:', train_labels.shape)
        print('Test data shape:', test_hist.shape)
        print('Test labels shape:', test_labels.shape)

    # plot the histogram of the first image
    plt.figure()
    plt.title(f'Histogram of the first image\nClass: {label2string(train_labels[0])}')
    plt.plot(train_hist[0])
    # add the axis labels
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    # add the label of the image class
    plt.show()
    # plot a figure with the histograms of the first 10 images
    plt.figure()
    plt.title('Histograms of the first 10 images')
    for i in range(10):
        plt.plot(train_hist[i], label=label2string(train_labels[i]))
        # add the axis labels
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    print('done')