import os
from sklearn.datasets import load_svmlight_file
import numpy as np

libsvm_root = '/data/libsvm/binary/'

libsvm_description = {
    'a9a': {'name': 'a9a', 'feature': 123, 'sample': 32561, 'label': [-1, 1]},
    'covtype': {'name': 'covtype', 'feature': 54, 'sample': 581012, 'label': [1, 2]},
    'diabetes': {'name': 'diabetes_scale', 'feature': 8, 'sample': 768, 'label': [-1, 1]},
    'german': {'name': 'german.numer_scale', 'feature': 24, 'sample': 1000, 'label': [-1, 1]},
    'gisette': {'name': 'gisette_scale', 'feature': 5000, 'sample': 6000, 'label': [-1, 1]},
    'ijcnn1': {'name': 'ijcnn1', 'feature': 22, 'sample': 141691, 'label': [-1, 1]},
    'real-sim': {'name': 'real-sim', 'feature': 20958, 'sample': 72309, 'label': [-1, 1]},
    'w8a': {'name': 'w8a', 'feature': 300, 'sample': 49749, 'label': [-1, 1]},
    'webspam_u': {'name': 'webspam_u', 'feature': 254, 'sample': 350000, 'label': [-1, 1]}
}

def libsvm_loader(filename):
    source = os.path.join(libsvm_root, libsvm_description[filename]['name'])
    data = load_svmlight_file(source)
    x_raw = data[0]
    y = np.array(data[1])
    # labels should be +1 or -1
    if libsvm_description[filename]['label'][0] == 1 and libsvm_description[filename]['label'][1] == 2:
        y = 2 * y - 3
    # add bias
    x = np.ones([x_raw.shape[0], x_raw.shape[1] + 1])
    x[:, :-1] = x_raw.todense()
    return x, y

if __name__ == '__main__':
    x, y = libsvm_loader('a9a')
    print((x.shape[0], x.shape[1] - 1))
