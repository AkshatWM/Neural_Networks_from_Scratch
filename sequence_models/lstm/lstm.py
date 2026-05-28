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

#LSTM class
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device = device)
        self.gates = nn.Linear(embedding_dim + hidden_size, 4 * hidden_size, device = device)
        self.lm_head = nn.Linear(hidden_size, vocab_size, device = device)
        self.hidden_size = hidden_size
    def forward(self, x, h_prev=None, c_prev=None):
        emb = self.embedding(x)
        B, T = x.shape
        
        if h_prev is None:
            h_prev = torch.zeros(B, self.hidden_size, device = x.device)
        if c_prev is None:
            c_prev = torch.zeros(B, self.hidden_size, device = x.device)
        
        logits_list = []
        for t in range(T):
            x_t = emb[:, t, :]

            combined = torch.cat([x_t, h_prev], dim=1)
            gates = self.gates(combined)

            f_t, i_t, c_hat_t, o_t = gates.chunk(4, dim=1)

            f_t = torch.sigmoid(f_t)
            i_t = torch.sigmoid(i_t)
            c_hat_t = torch.tanh(c_hat_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_prev + i_t * c_hat_t
            h_t = o_t * torch.tanh(c_t)

            logits = self.lm_head(h_t)
            logits_list.append(logits)

            h_prev = h_t
            c_prev = c_t
        
        logits = torch.stack(logits_list, dim = 1)
        return logits, (h_t, c_t)
    
#Adjustable knobs
vocab_size = 1024
embedding_dim = 42
hidden_size = 150
batch_size = 32
lr = 0.001
steps = 5000 

#Instantiating the model
model = LSTMLanguageModel(vocab_size, embedding_dim, hidden_size)
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

        row, _ = x_dataset.shape
        
        idx = torch.randint(0, row, (batch_size,))
        
        x_batch = x_dataset[idx]
        y_batch = y_dataset[idx]
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        B, T = x_batch.shape
        
        logits, _ = model(x_batch)

        loss = F.cross_entropy(logits.reshape(B*T, vocab_size), y_batch.reshape(B*T))
        
        return loss.item()

#Training loop
for i in range(steps):

    model.train() #For batch norm (if used)
    #Forward pass
    #sample batch
    row, _ = x_train.shape
    idx = torch.randint(0, row, (batch_size,))

    x_batch = x_train[idx]
    y_batch = y_train[idx]

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    B, T = x_batch.shape

    logits, _ = model(x_batch)
    loss = F.cross_entropy(logits.reshape(B*T, vocab_size), y_batch.reshape(B*T))

    #Backward pass
    optimizer.zero_grad()
    loss.backward()

    #Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    #Update
    optimizer.step()

    #Print loss
    if i%100 == 0:
        train_loss = get_loss("train")
        val_loss = get_loss("test")
        print(f"Step:{i}/{steps} train loss:{train_loss:.4f} test_loss:{val_loss:.4f}")

#Sampling from the model
model.eval()
with torch.no_grad():
    max_tokens = 300
    token = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    h_prev = None
    c_prev = None

    generated = []

    for _ in range(max_tokens):
        logits, (h_prev, c_prev) = model(token, h_prev, c_prev)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples= 1)
        generated.append(token.item())

    output = tokenizer.decode(generated)
    print(output)