# Removes the one-hot encoding as a step performed on xs
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

# create a classifier class that inherits from nn.Module
class BigramClassifier(torch.nn.Module):
    def __init__(self):
        super(BigramClassifier, self).__init__()
        self.W = torch.nn.Parameter(torch.randn((27,27), generator=g, requires_grad=True))

    # x here is no longer a one-hot encoded vector, instead we must select the row of W that corresponds to the index of the letter
    def forward(self, x):
        return self.W[x]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

g = torch.Generator().manual_seed(42)

words = open('names.txt').read().splitlines()
letters = sorted(list(set(''.join(words))))
letter_to_index = {letter: index for index, letter in enumerate(letters)}
letter_to_index['.'] = 0
index_to_letter = {i: letter for letter, i in letter_to_index.items()}

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(letter_to_index[ch1])
        ys.append(letter_to_index[ch2])

xs = torch.as_tensor(xs)
ys = torch.as_tensor(ys)

dataset = TensorDataset(xs, ys)

train_ratio = .8
validation_ratio = .1

n_total = len(dataset)
n_train = int(n_total * train_ratio)
n_train_batch=n_train
n_validation = int(n_total * validation_ratio)
n_validation_batch=n_validation
n_test = n_total - n_train - n_validation

train_data, validation_data, test_data = random_split(dataset, [n_train, n_validation, n_test])

train_loader = DataLoader(train_data, batch_size=n_train_batch, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=n_validation_batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size=n_test, shuffle=True)
