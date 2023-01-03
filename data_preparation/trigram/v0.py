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

for w in words:
    chs = ['.'] + list(w) + ['.']
    # create trigrams from the words
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        xs_train.append((letter_to_index[ch1], letter_to_index[ch2]))
        ys_train.append(letter_to_index[ch3])

