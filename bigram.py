import torch
import torch.nn as nn
from torch.nn import functional as F

dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # because we are not using backward, makes pytorch faster
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model (X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ One Head of Self-Attention"""
    def __init__(self, block_size, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, block_size, n_embd, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, block_size, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(block_size, n_embd, n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size = 26, block_size = 16, n_embd = 32, n_head = 4, n_layer = 6):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(self.block_size, n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:] # because no targets -> Loss is None -> logits still have to be permuted -> becomes (B, C)
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1) # Es gibt ja jetzt für jeden Token eine softmax wahrscheinlichkeit
            # torch.multinomial nimmt jetzt zufällig einen dieser Token with respect to their wahrscheinlichkeit
            # es ist also am wahrscheinlichsten, dass der Token mit der höchsten probability genommen wird, aber es können auch
            # unwahrscheinlichere gewählt werden, was die Ausgabe natürlicher macht
            # zum beispiel: "Die Katze liegt auf dem " -> Sofa, Bett, Boden -> Greedy würde immer das selbe wählen
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 8
    block_size = 32  # Context Length
    max_iters = 1000
    eval_interval = 500
    learning_rate = 3e-4 #1e-3
    eval_iters = 200
    n_embd = 64 # needs to be divisible by n_head
    n_head = 8
    n_layer = 6
    save_path = 'model_exp.pth'
    # --------------------
    torch.manual_seed(1337)

    with open('news-commentary-v10.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
    itos = {i: ch for i, ch in enumerate(chars)}  # integer to string
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    config = {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer
    }

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel(**config)
    m = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush = True)

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1,1), dtype = torch.long, device = device)
    print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config
    }

    torch.save(checkpoint, save_path)
