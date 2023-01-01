import torch
import torch.optim as optim
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

g = torch.Generator().manual_seed(42)

words = open('names.txt').read().splitlines()
letters = sorted(list(set(''.join(words))))
letter_to_index = {letter: index for index, letter in enumerate(letters)}
letter_to_index['.'] = 0
index_to_letter = {i: letter for letter, i in letter_to_index.items()}

xs_train, ys_train = [], []
xs_train_num = 0

validation_words, test_words = [], []

trainValTestSplit = [0.8, 0.1, 0.1]

indices = torch.randperm(len(words), generator=g)

for i in indices[:int(len(indices) * trainValTestSplit[0])]:
    chs = ['.'] + list(words[i]) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs_train.append(letter_to_index[ch1])
        ys_train.append(letter_to_index[ch2])

for i in indices[int(len(indices) * trainValTestSplit[0]):int(len(indices) * (trainValTestSplit[0] + trainValTestSplit[1]))]:
    validation_words.append(words[i])

for i in indices[int(len(indices) * (trainValTestSplit[0] + trainValTestSplit[1])):]:
    test_words.append(words[i])    


xs_train = torch.tensor(xs_train).to(device)
ys_train = torch.tensor(ys_train).to(device)

xs_train_num = xs_train.nelement()
xs_train_encoded = torch.nn.functional.one_hot(xs_train, num_classes=27).float()

# create a classifier class that inherits from nn.Module
class BigramClassifier(torch.nn.Module):
    def __init__(self):
        super(BigramClassifier, self).__init__()
        self.W = torch.nn.Parameter(torch.randn((27,27), generator=g, requires_grad=True))

    def forward(self, x):
        return x @ self.W