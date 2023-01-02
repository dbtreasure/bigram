import torch
import torch.nn.functional as F

def generate(count):
    for i in range(count):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ model.W # predict log-counts
            counts = logits.exp() # counts, equivalent to N
            p = counts / counts.sum(1, keepdims=True) # probabilities for next character

            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(index_to_letter[ix])
            if ix == 0:
                break
        print(''.join(out[:-1]))

generate(5)