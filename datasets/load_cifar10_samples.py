import os.path


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_samples():
    import os
    import numpy as np
    data_dir = os.path.join(os.getcwd(), 'cifar-10-batches-py')
    train_data = None
    train_labels = []
    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic[b'data']
        else:
            train_data = np.vstack((train_data, data_dic[b'data']))
        train_labels += data_dic[b'labels']
    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic[b'data']
    test_labels = test_data_dic[b'labels']
    return train_data, train_labels, test_data, test_labels

def save_samples_as_images():
    import matplotlib.pyplot as plt
    data_dir = os.path.join(os.getcwd(), 'cifar-10-batches-py')
    for i in range(1, 2):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        for j in range(10000):
            img = data_dic[b'data'][j]
            img = img.reshape(3, 32, 32).transpose(1, 2, 0)
            plt.imsave(data_dir + '/images/{}_{}.png'.format(data_dic[b'labels'][j], j), img)
            if j == 10:
                break

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_cifar10_samples()
    print(train_data.shape)
    print(len(train_labels))
    print(test_data.shape)
    print(len(test_labels))
    # save_samples_as_images()