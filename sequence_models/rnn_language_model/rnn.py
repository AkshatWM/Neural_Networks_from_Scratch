#Imports
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

#Loading my BPE tokenizer package
from my_tokenizer.gpt_bpe_tokenizer.serialization import load_tokenizer
from my_tokenizer.gpt_bpe_tokenizer.regex_tokenizer import RegexTokenizer

#Loading the dataset
dataset_path = PROJECT_ROOT / "datasets" / "tinyphilosopher.txt"
with open(dataset_path, 'r', encoding = 'utf-8') as f:
    text = f.read()


tokenizer = load_tokenizer(RegexTokenizer, PROJECT_ROOT / "checkpoints" / "tokenizer" / "gpt4tokenizer.model")

#Encoding the dataset
ids = torch.tensor(tokenizer.encode(text))

#Splitting the dataset into train and test dataset
dataset_size = len(ids)
train_size = int(0.9 * dataset_size)
x_train , x_test= ids[:train_size], ids[train_size:]
y_train, y_test = ids[1:train_size+1], ids[train_size+1:]

#Defining the time dimension aka the no. of tokens in a sequence
T = 8
#Preparing dataset for processing
seq_len = (len(x_train) // T) * T
seq_len2 = (len(x_test) // T) * T
x_train = x_train[:seq_len].view(-1, T)
y_train = y_train[:seq_len].view(-1, T)
x_test = x_test[:seq_len2].view(-1, T)
y_test = y_test[:seq_len2].view(-1, T)

torch.manual_seed(42)
#Additional optimization for GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#RNN class
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.W_xh = nn.Linear(embedding_dim, hidden_size, bias=True, device=device)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True, device=device)        
        self.W_hy = nn.Linear(hidden_size, vocab_size, bias=True, device=device)
        self.embd = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, T = x.shape
        embds = self.embd(x)
        h_prev = torch.zeros(batch_size, self.hidden_size, device = x.device)

        outputs = []

        for t in range(T):

            x_t = embds[:, t, :]
            h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))
            y_t = self.W_hy(h_t)
            h_prev = h_t
            outputs.append(y_t)
        
        logits = torch.stack(outputs, dim = 1)
        return logits

#Adjustable knobs
vocab_size = 1024
embedding_dim = 42
hidden_size = 150
batch_size = 32
lr = 0.001
epochs = 5000 

#Instantiating the model
model = RNNLanguageModel(vocab_size= vocab_size, embedding_dim= embedding_dim, hidden_size= hidden_size)
optimizer = optim.AdamW(model.parameters(), lr=lr)

#Function to get loss for both train and test dataset
def get_loss(split):

    if split == "train":
        model.train()
        x_dataset, y_dataset = x_train, y_train
    else:
        model.eval()
        x_dataset, y_dataset = x_test, y_test

    with torch.no_grad():

        # sample batch
        row, _ = x_dataset.shape

        idx = torch.randint(0, row, (batch_size,))

        x_batch = x_dataset[idx]
        y_batch = y_dataset[idx]

        # forward pass
        logits = model(x_batch)

        # loss
        B, T, C = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B * T, C),
            y_batch.reshape(B * T)
        )

    return loss.item()

#Training loop
for i in range(epochs):
    #Forward pass
    #sample batch
    row, T = x_train.shape
    idx = torch.randint(0, row, (batch_size,))

    x_batch, y_batch = x_train[idx], y_train[idx]
    #embds
    logits = model(x_batch)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.reshape(B*T,C), y_batch.reshape(B*T))

    #Backward pass
    optimizer.zero_grad()
    loss.backward()
    #gradient clipping to prevent exploding gradients
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    #Update
    optimizer.step()

    #Print loss
    if i%100 == 0:
        train_loss = get_loss("train")
        val_loss = get_loss("test")
        print(f"Epochs:{i}/{epochs} train loss:{train_loss:.4f} test_loss:{val_loss:.4f}")

#Sampling from the model
model.eval()
with torch.no_grad():
    max_tokens = 100

    token = torch.zeros((1, 1), dtype=torch.long, device=device)
    h_prev = torch.zeros((1, hidden_size), device=device)

    generated = []

    for _ in range(max_tokens):

        embds = model.embd(token)
        x_t = embds[:, 0, :]

        h_t = torch.tanh(model.W_xh(x_t) + model.W_hh(h_prev))
        y_t = model.W_hy(h_t)
        h_prev = h_t

        probs = F.softmax(y_t, dim=-1)
        token = torch.multinomial(probs,num_samples=1)

        generated.append(token.item())

    # decode generated ids
    output = tokenizer.decode(generated)

    print(output)