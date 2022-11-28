import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

# TODO: add support for bw images

# Load dataset and split in train, validation and test sets (with different transformations)
class ImageDataset(Dataset):
    
    def __init__(self, labels, images, targets, transform=None):
        self.labels = labels
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.labels)}
        self.images = images
        self.targets = targets
        self.transform = transform
    
    def flow_from_directory(root_dir, transform=None, shuffle=False, verbose=False):
        # Load labels from folders
        labels = os.listdir(root_dir)
        labels.sort()
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Load images and labels
        images = []
        targets = []
        for label in labels:
            for image in os.listdir(os.path.join(root_dir, label)):
                images.append(os.path.join(root_dir, label, image))
                targets.append(label_to_idx[label])
                
        transform = transform
        
        # Create dataset
        dataset = ImageDataset(labels, images, targets, transform)
        
        # Shuffle data
        if shuffle:
            dataset.shuffle()
            
        # Print dataset info
        if verbose:
            print('Dataset info:')
            print('Number of samples: {}'.format(len(dataset.images)))
            print('Number of labels: {}'.format(len(dataset.labels)))
            print('Labels: {}'.format(dataset.labels))
            print()
            
        return dataset
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        target = self.targets[idx]
        
        # Apply transformation to sample (if any)
        if self.transform:
            image = self.transform(image)
            
        return image, target
    
    # Shuffle data
    def shuffle(self):
        idx = np.arange(len(self.images))
        np.random.shuffle(idx)
        self.images = [self.images[i] for i in idx]
        self.targets = [self.targets[i] for i in idx]
        
    # Split dataset in train, validation and test sets
    def split(self, eval_split, test_split, train_transform=None, eval_transform=None, test_transform=None, verbose=False):
        # Compute number of samples for each set
        n_samples = len(self.images)
        n_eval = int(n_samples * eval_split)
        n_test = int(n_samples * test_split)
        n_train = n_samples - n_eval - n_test
        
        # Split data
        train_dataset = ImageDataset(self.labels, self.images[:n_train], self.targets[:n_train], train_transform)
        eval_dataset = ImageDataset(self.labels, self.images[n_train:n_train+n_eval], self.targets[n_train:n_train+n_eval], eval_transform)
        test_dataset = ImageDataset(self.labels, self.images[n_train+n_eval:], self.targets[n_train+n_eval:], test_transform)
        
        # Print dataset info
        if verbose:
            # print divider
            print('Train set info:')
            print('Number of samples: {}'.format(len(train_dataset.images)))
            print('Number of labels: {}'.format(len(train_dataset.labels)))
            print('-' * 30)
            print('Validation set info:')
            print('Number of samples: {}'.format(len(eval_dataset.images)))
            print('Number of labels: {}'.format(len(eval_dataset.labels)))
            print('-' * 30)
            print('Test set info:')
            print('Number of samples: {}'.format(len(test_dataset.images)))
            print('Number of labels: {}'.format(len(test_dataset.labels)))
        
        return train_dataset, eval_dataset, test_dataset
        
    
# Function to show some images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()