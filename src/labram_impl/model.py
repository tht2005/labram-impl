import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import calculate_channel_phase

class TemporalEncoder(nn.Module):
    def __init__(self, in_chan=1, out_chan=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gn1 = nn.GroupNorm(4, out_chan)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.gn2 = nn.GroupNorm(4, out_chan)
        self.gelu2 = nn.GELU()
        self.conv3 = nn.Conv2d(out_chan, out_chan, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.gn3 = nn.GroupNorm(4, out_chan)
        self.gelu3 = nn.GELU()
    def forward(self, x):
        # B, N, A, T = x.shape
        x = rearrange(x, 'B N A T -> B (N A) T')
        x = x.unsqueeze(1)
        x = self.gelu1(self.gn1(self.conv1(x)))
        x = self.gelu2(self.gn2(self.conv2(x)))
        x = self.gelu3(self.gn3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        # x = rearrange(x, 'B (N A) T -> B N A T', N=N, A=A)
        return x

class Codebook(nn.Module):
    def __init__(self, n_embd=8192, d_embd=64):
        super().__init__()
        self.embd = nn.Embedding(n_embd, d_embd)

class AttentionHead(nn.Module):
    def __init__(self, n_embd, d_embd, d_hid):
        super().__init__()
        self.wQ = nn.Linear(d_embd, d_hid)
        self.wK = nn.Linear(d_embd, d_hid)
        self.wV = nn.Linear(d_embd, d_hid)
        self.lnQ = nn.LayerNorm((d_hid))
        self.lnK = nn.LayerNorm((d_hid))
        self.scale = d_hid ** -0.5
    def forward(self, x):
        Q = self.lnQ(self.wQ(x))
        K = self.lnK(self.wK(x))
        V = self.wV(x)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        attn = torch.matmul(attn, V)
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embd, d_embd, d_hid=200):
        super().__init__()
        self.n_head = n_head
        # input (B, n_embd, d_embd) => (B, n_embd, d_hid)
        self.heads = nn.ModuleList([
            AttentionHead(n_embd, d_embd, d_hid) for _ in range(n_head)
        ])
        # cat: (B, n_embd, n_head * d_hid)
        self.wO = nn.Linear(n_head * d_hid, d_embd)
    def forward(self, x):
        y = [ f(x) for f in self.heads ]
        w = torch.cat(y, dim=-1)
        return self.wO(w)

class MLP(nn.Module):
    def __init__(self, d_embd, d_hid=800):
        super().__init__()
        self.ln1 = nn.Linear(d_embd, d_hid)
        self.relu = nn.ReLU()
        self.ln2 = nn.Linear(d_hid, d_embd)
    def forward(self, x):
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, n_embd, d_embd, n_head=10):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, n_embd, d_embd)
        self.norm1 = nn.LayerNorm((d_embd))
        self.mlp = MLP(d_embd)
        self.norm2 = nn.LayerNorm((d_embd))
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x

class NeuralTokenizerTrainer(nn.Module):
    def __init__(self, n_embd, d_embd, d_codebook, n_trans_enc=12, n_trans_dec=3):
        super().__init__()
        self.enc = nn.ModuleList([ Transformer(n_embd, d_embd) for _ in range(n_trans_enc) ])
        self.codebook = nn.Embedding(d_codebook, d_embd);
        self.dec = nn.ModuleList([ Transformer(n_embd, d_embd) for _ in range(n_trans_dec) ])
    def l2(self, x):
        return F.normalize(x, dim=-1)
    def square_norm(self, x):
        return (x*x).sum(dim=-1)
    def forward(self, x):
        x_label = x
        for f in self.enc: x = f(x)
        x = self.l2(x)
        C = self.l2(self.codebook.weight)
        dist = (x ** 2).sum(dim=-1, keepdim=True)               \
            + (C ** 2).sum(dim=-1)                              \
            - 2 * torch.matmul(x, C.transpose(-2, -1))
        indices = dist.argmin(dim=-1)
        x_q = self.codebook(indices)
        tmp = x_q
        for f in self.dec: tmp = f(tmp)

        (A, Phi) = calculate_channel_phase(tmp)
        (A_label, Phi_label) = calculate_channel_phase(x_label) 
        return self.square_norm(A - A_label)                                \
                + self.square_norm(Phi - Phi_label)                         \
                + self.square_norm(x.detach() - self.codebook(indices))     \
                + self.square_norm(x - self.codebook(indices).detach())

