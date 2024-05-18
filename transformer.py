import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, n_channels=1, embed_dim=96, img_size=28, patch_size=4):
        super().__init__()
        self.patch = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim)) # <cls> token
        self.pos = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2 + 1, embed_dim))
    def forward(self, x):
        x = self.patch(x) # B C H W -> B E H/4 W/4
        x = x.flatten(2).transpose(1, 2) # B E H/4 W/4 -> B H/4*W/4 E
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1) # B H/4*W/4 E -> B H/4*W/4+1 E
        x = x + self.pos
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim=96):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)

    # TODO: try vision transformer with different sequence
    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.act = nn.Tanh()
    def forward(self, x):
        x = x[:, 0, :] # get <cls> token
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim=128, n_layers=6, n_class=10):
        super().__init__()
        self.embedding = Embedding(embed_dim=embed_dim)
        self.encoder = nn.Sequential(
            *[Encoder(embed_dim) for _ in range(n_layers)],
            nn.LayerNorm(embed_dim))
        self.mlp = MLP(embed_dim, n_class)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.mlp(x)
        return x
        