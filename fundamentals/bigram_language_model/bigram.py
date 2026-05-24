#importing libraries
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
with open('names.txt','r') as f:
    words = f.read().split()

#Integer to Character mapping and the opposite 
itos = {i+1:j for i,j in enumerate(sorted(list(set("".join(words)))))}
itos[0]='.'
stoi = {j:i for i,j in itos.items()}

#Counting which character follows which for the entire dataset
n=2
count = {}
for w in words:
    word = ['.']*(n-1) + list(w) + ['.']
    for i in range(len(word)-n+1):
        context = tuple(word[i:n+i-1])
        target = word[n+i-1]
        count[(context,target)] = count.get((context,target), 0) + 1

vocab_size = len(stoi)
N = np.zeros((vocab_size, vocab_size))

for i,j in count.items():
    row = stoi[i[0][0]]
    col = stoi[i[1]]
    N[row,col] += j

N = N + 1
P = N/ N.sum(axis=1, keepdims=True) 

loss = 0
total = 0
for (context, target), cnt in count.items():
    row = stoi[context[0]]
    col = stoi[target]
    loss += cnt * np.log(P[row,col])
    total += cnt  

loss = -loss/total

print(f"{loss=}")

s = stoi['.']
for i in range(10):
    new = []
    char = ""
    while (char!='.'):
        idx = np.random.choice(vocab_size, p=P[s])
        char = itos[idx]
        s = idx
        new.append(char)
    print(''.join(new))