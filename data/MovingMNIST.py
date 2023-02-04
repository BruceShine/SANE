import os
import numpy as np
import torch
import torch.utils.data as data


def load_moving_mnist(file_dir, is_train):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(file_dir, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    if is_train:
        dataset = dataset[:, :8000, :]
        length = 8000
    else:
        dataset = dataset[:, 8000:, :]
        length = 2000
    return dataset, length


class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output, transform=None):
        super(MovingMNIST, self).__init__()

        self.dataset, self.length = load_moving_mnist(root, is_train)

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output

    def __getitem__(self, idx):
        images_array = self.dataset[:, idx, ...]
        input_array = images_array[:self.n_frames_input]
        target_array = images_array[self.n_frames_input:]

        input_seq = torch.from_numpy(input_array / 255.0).contiguous().float()
        input_seq = input_seq.permute(0, 3, 1, 2)
        target_seq = torch.from_numpy(target_array / 255.0).contiguous().float()
        target_seq = target_seq.permute(0, 3, 1, 2)

        return input_seq, target_seq

    def __len__(self):
        return self.length
