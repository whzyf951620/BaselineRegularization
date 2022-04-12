import os
from cv2 import split
import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import math
from LinearRegionSplit.ellipse import get_fitted_ellipse_and_stats
import numpy as np
import numexpr as ne


ROOT = './MNIST'
train_images = 'train-images-idx3-ubyte'
train_labels = 'train-labels-idx1-ubyte'
test_images = 't10k-images-idx3-ubyte'
test_labels = 't10k-labels-idx1-ubyte'

filelist = [train_images, train_labels, test_images, test_labels]

def Read_Dataset(root, filelist):
    train_images, train_labels, test_images, test_labels = filelist
    train_images_path = os.path.join(ROOT, train_images)
    with open(train_images_path, 'rb') as f:
        data = f.read(16)
        des, img_nums, row, col = struct.unpack_from('>IIII', data, 0)
        train_x = np.zeros((img_nums, row * col))
        for index in range(img_nums):
            data = f.read(784)
            if len(data) == 784:
                train_x[index, :] = np.array(struct.unpack_from('>' + 'B' * (row * col), data, 0)).reshape(1, 784)
        f.close()

    train_labels_path = os.path.join(ROOT, test_labels)
    with open(train_labels_path, 'rb') as f:
        data = f.read(8)
        des,label_nums = struct.unpack_from('>II', data, 0)
        train_y = np.zeros((label_nums, 1))
        for index in range(label_nums):
            data = f.read(1)
            train_y[index,:] = np.array(struct.unpack_from('>B', data, 0)).reshape(1,1)
        f.close()

    return {'train_x': train_x, 'train_y': train_y}

def select_3vertices(train_y, label = -1, label_type = 'same'):
    label_unique = np.unique(train_y).astype(int)
    if label_type == 'same':
        tmp = np.full_like(train_y, label)
        for index, item in enumerate(label_unique):
            if item == label:
                indices, _ = np.nonzero(np.equal(train_y, tmp))
                break

        num1, num2, num3 = indices[:3]
    elif label_type == 'diff':
        label_diff = np.random.choice(label_unique, 3, replace = False)
        tmp0 = np.full_like(train_y, label_diff[0])
        tmp1 = np.full_like(train_y, label_diff[1])
        tmp2 = np.full_like(train_y, label_diff[2])
        tmp = [tmp0, tmp1, tmp2]
        nums = []
        for index, label_select in enumerate(label_diff):
            indices, _ = np.nonzero(np.equal(train_y, tmp[index]))
            tmp_num = np.random.randint(len(indices))
            nums.append(indices[tmp_num])
            
        num1, num2, num3 = nums
    return [num1, num2, num3]

def yield_samples_on_2Dplane(data_dict, label = 4, Num = 10000, label_type = 'same'):
    train_x, train_y = data_dict['train_x'], data_dict['train_y']
    num1, num2, num3 = select_3vertices(train_y, label = label, label_type = label_type)
    img1 = np.expand_dims(train_x[num1], 0)
    img2 = np.expand_dims(train_x[num2], 0)
    img3 = np.expand_dims(train_x[num3], 0)
    A = np.tile(img1, (Num, 1))
    B = np.tile(img2, (Num, 1))
    C = np.tile(img3, (Num, 1))
    s = np.linspace(0, 1, int(math.sqrt(Num)))
    t = np.linspace(0, 1, int(math.sqrt(Num)))
    s, t = np.meshgrid(s, t)
    s, t = s.reshape(-1), t.reshape(-1)
    s = np.tile(np.expand_dims(s, 1), (1, 784))
    t = np.tile(np.expand_dims(t, 1), (1, 784))
    D = s * B + t * C + A
    return D

class MNIST_2D_plane(Dataset):
    def __init__(self, Num = 10000, label_type = 'same'):
        super(MNIST_2D_plane, self).__init__()
        data_dict = Read_Dataset(ROOT, filelist)
        if label_type == 'same':
            label = 4
        else:
            label = -1
        self.data = yield_samples_on_2Dplane(data_dict, label = label, Num = Num, label_type = label_type)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
		
class EllipseDataset(Dataset):
    def __init__(self, features, Num = 10000, label_type = 'same', split_num = 100):
        super(EllipseDataset, self).__init__()
        data_dict = dataset
        if label_type == 'same':
            label = 4
        else:
            label = -1
            
        self.vertices = None
        self.transform = None
        self.rads = None
        self.Num = Num
        self.split_num = split_num
        self.nums = select_3vertices(data_dict['train_y'], label, label_type)
        self.vertices = self._yield_vertices(self.nums, data_dict['train_x'])
        
    def get_data(self, index):
        self.data = self.get_fitted_ellipse_and_stats(self.Num, self.split_num, index)

    def _yield_vertices(self, nums, train_x):
        if self.vertices is None:
            num1, num2, num3 = nums
            img1 = np.expand_dims(train_x[num1], 0)
            img2 = np.expand_dims(train_x[num2], 0)
            img3 = np.expand_dims(train_x[num3], 0)
            vertices = np.concatenate([img1, img2, img3], axis = 0)
        else:
            pass
        return vertices

    def _get_zero_centered_circle(self, radius: float, num_samples: int, split_num: int, index: int) -> np.ndarray:
        """Get a zero-centered circle of a given radius.
        Args:
        radius: radius of the circle.
        num_samples: number of points to sample on the circle.
        Returns:
        A numpy array of circle points.
        """
        assert num_samples % split_num == 0
        each = num_samples // split_num
        if self.rads is None:
            self.rads = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        rads_each = self.rads[index * each: (index + 1) * each]
        # circle = np.zeros([num_samples, 2])
        circle = np.zeros([each, 2])
        ne.evaluate('cos(rads_each)', out=circle[:, 0])
        ne.evaluate('sin(rads_each)', out=circle[:, 1])
        circle *= radius
        return circle

    def _get_2d_points_fitting_transform(self, vertices: np.ndarray, 
                                     scale: float) -> np.ndarray:
        """Get a matrix transformation projecting circle points onto vertices.
        Args:
            vertices: resulting points of the transformation.
            scale: scale (radius) of the input equidistant circle points.
        Returns:
            Transformation matrix.
        """
        # angles = np.pi * np.linspace(0, 2, len(vertices), endpoint=False)
        angles = np.array([np.pi / 3, np.pi, np.pi * 5 / 3])
        angles = np.reshape(angles, (-1, 1))
        coords_2d = np.concatenate((np.cos(angles), np.sin(angles)), axis=1) * scale
        self.transform = np.dot(np.linalg.pinv(coords_2d), vertices)
        return self.transform


    def get_fitted_ellipse_and_stats(self, num_samples: int, split_num: int, index: int) -> np.ndarray:
        """Get an ellipse passing through given vertices.
        The ellipse is centered at the mean of the vertices;
        Args:
            num_samples: number of points sampled on the circle.
            vertices: vertices to fit the circle to.
        Returns:
            A numpy array of points on a circle and dict with statistics.
        """
        shape = self.vertices[0].shape if self.vertices.ndim > 1 else (1,)
        self.vertices = np.reshape(self.vertices, (self.vertices.shape[0], -1))

        center = np.mean(self.vertices, axis=0)
        circle = self._get_zero_centered_circle(1, num_samples, split_num, index)

        if self.transform is None:
            self.transform = self._get_2d_points_fitting_transform(self.vertices - center, scale=1)
        circle = np.dot(circle, self.transform)
        circle += center
        circle = np.reshape(circle, (num_samples // split_num,) + shape)
        return circle

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    pass
