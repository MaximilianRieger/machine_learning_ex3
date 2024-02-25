# create the histograms for the cifar10 dataset

from datasets.load_celebA_samples import load_celebA_samples
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os

# create rgb histogram
def create_rgb_histograms(data):
    histograms = []
    for img_path in tqdm.tqdm(data, desc='Creating histograms', unit='image'):
        img = plt.imread(img_path)
        red_hist = np.histogram(img[:,:,0].ravel(), bins=256, range=[0, 256])[0]
        green_hist = np.histogram(img[:,:,1].ravel(), bins=256, range=[0, 256])[0]
        blue_hist = np.histogram(img[:,:,2].ravel(), bins=256, range=[0, 256])[0]
        histograms.append(np.concatenate((red_hist, green_hist, blue_hist)))
    return np.array(histograms)

def label2string(label):
    labels = ['Smiling', 'Blond_Hair', 'Black_Hair', 'Brown_Hair', 'Gray_Hair', 'Pale_Skin', 'Narrow_Eyes', 'Rosy_Cheeks', 'Eyeglasses', 'Wearing_Lipstick']
    return labels[label]

if __name__ == '__main__':
    # check if the histograms are already created and load them
    if not os.path.exists(os.path.join(os.getcwd(), 'train_hist.npy')):
        print('Creating histograms...')
        # load the celebA dataset
        train_data, train_labels, test_data, test_labels = load_celebA_samples()
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
    hists = train_hist[::5000]
    labels = train_labels[::5000]
    for i, data in enumerate(zip(hists, labels)):
        hist, label = data
        plt.plot(hist, label=label2string(label))
        # add the axis labels
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    print('done')