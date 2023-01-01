import torch

class BigramModel():
    # initialize your data structures in the constructor
    def __init__(self, corpus, generator, trainValTestSplit = [0.8, 0.1, 0.1]):
        self.corpus = open(corpus, 'r').read().splitlines()
        self.letters = sorted(list(set(''.join(self.corpus))))
        self.trainValTestSplit = trainValTestSplit
        self.generator = generator
        self.indices = torch.randperm(len(self.corpus), generator=self.generator)

        self.letter_to_index = {letter: index for index, letter in enumerate(self.letters)}
        self.letter_to_index['.'] = 0
        self.index_to_letter = {i: letter for letter, i in self.letter_to_index.items()}
        
        self.W = torch.randn((27,27), generator=self.generator, requires_grad=True)

        self.xs_train, self.ys_train, self.valid_words, self.test_words = [], [], [], []

        self.setup_train()
        self.setup_valid()
        self.setup_test()

    def setup_train(self):
        for i in self.indices[:int(len(self.indices) * self.trainValTestSplit[0])]:
            chs = ['.'] + list(self.corpus[i]) + ['.']
            # iterate through the first and second characters for a zip of the characters, 
            # and the characters after the first character
            for ch1, ch2 in zip(chs, chs[1:]):
                # append the index of the first character to the xs_train variable
                self.xs_train.append(self.letter_to_index[ch1])
                # append the index of the second character to the ys_train variable
                self.ys_train.append(self.letter_to_index[ch2]) 
        
        self.xs_train = torch.tensor(self.xs_train)
        self.ys_train = torch.tensor(self.ys_train)
        self.xs_train_num = self.xs_train.nelement()
        self.xs_train_encoded = torch.nn.functional.one_hot(self.xs_train, num_classes=27).float()

    def setup_valid(self):
        for i in self.indices[int(len(self.indices) * self.trainValTestSplit[0]):int(len(self.indices) * (self.trainValTestSplit[0] + self.trainValTestSplit[1]))]:
            self.valid_words.append(self.corpus[i])

    def setup_test(self):
        for i in self.indices[int(len(self.indices) * (self.trainValTestSplit[0] + self.trainValTestSplit[1])):]:
            self.test_words.append(self.corpus[i])

    def train_test(self, epochs):
        for epoch in range(epochs):
            log_counts = self.xs_train_encoded @ self.W
            exp_counts = log_counts.exp()
            probz = exp_counts / exp_counts.sum(dim=1, keepdim=True)

            lossz = -probz[torch.arange(self.xs_train_num), self.ys_train].log().mean() + 0.01 * self.W.pow(2).mean()
            print(lossz.item())

            self.W.grad = None
            lossz.backward()

            self.W.data += -50 * self.W.grad
    
    def generate(self, count):
        for i in range(5):
            out = []
            ix = 0
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                logits = xenc @ W # predict log-counts
                counts = logits.exp() # counts, equivalent to N
                p = counts / counts.sum(1, keepdims=True) # probabilities for next character
                # ----------
                
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(index_to_letter[ix])
                if ix == 0:
                    break
            print(''.join(out[:-1]))