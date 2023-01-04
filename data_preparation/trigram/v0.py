import torch
import torch.optim as optim
import torch.nn.functional as F

# create a classifier class that inherits from nn.Module
class TrigramClassifier(torch.nn.Module):
    def __init__(self):
        super(TrigramClassifier, self).__init__()
        self.W = torch.nn.Parameter(torch.randn((27,27,27), generator=g, requires_grad=True))

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

trigram_xs_train, trigram_ys_train = [], []

trigram_validation_words, trigram_test_words = [], []

trigram_trainValTestSplit = [0.8, 0.1, 0.1]

indices = torch.randperm(len(words), generator=g)

for w in words:
    chs = ['.'] + list(w) + ['.']
    # create trigrams from the words
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        xs_train.append((letter_to_index[ch1], letter_to_index[ch2]))
        ys_train.append(letter_to_index[ch3])

trigram_xs = torch.as_tensor(xs)
trigram_ys = torch.as_tensor(ys)

trigram_dataset = TensorDataset(xs, ys)

trigram_train_ratio = .8
trigram_validation_ratio = .1

trigram_n_total = len(dataset)
trigram_n_train = int(n_total * train_ratio)
trigram_n_train_batch=n_train
trigram_n_validation = int(n_total * validation_ratio)
trigram_n_validation_batch=n_validation
trigram_n_test = n_total - n_train - n_validation

trigram_train_data, trigram_validation_data, trigram_test_data = random_split(dataset, [n_train, n_validation, n_test])

trigram_train_loader = DataLoader(train_data, batch_size=n_train_batch, shuffle=True)
trigram_validation_loader = DataLoader(validation_data, batch_size=n_validation_batch, shuffle=True)
trigram_test_loader = DataLoader(test_data, batch_size=n_test, shuffle=True)