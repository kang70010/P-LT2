import torch
import torch.nn as nn
import torch.fft as fft

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()


        x = self.conv1(x)
        x = self.conv2(x)
        _, _, h, w = x.shape

        if h % self.ph != 0 or w % self.pw != 0:

            pad_h = (self.ph - h % self.ph) % self.ph
            pad_w = (self.pw - w % self.pw) % self.pw
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, h, w = x.shape
        x = x.reshape(-1, self.ph, self.pw, h // self.ph, w // self.pw, x.shape[1])
        x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
        x = x.view(-1, self.ph * self.pw, x.shape[-1])

        x = self.transformer(x)

        x = x.view(-1, h // self.ph, w // self.pw, self.ph, self.pw, x.shape[-1])
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(-1, x.shape[1], h, w)

        if x.shape[2] != y.shape[2] or x.shape[3] != y.shape[3]:
            x = x[:, :, :y.shape[2], :y.shape[3]]

        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class FSCABlock(nn.Module):

    def __init__(self, c1, c2, patch_size=2, transformer_depth=2, mlp_dim=512):
        super().__init__()
        assert c1 == c2,
        self.c1 = c1
        self.c2 = c2

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        self.freq_weight = nn.Parameter(torch.ones(1, c1, 1, 1) * 0.5)

        self.mobilevit = MobileViTBlock(
            dim=c1,
            depth=transformer_depth,
            channel=c1,
            kernel_size=3,
            patch_size=patch_size,
            mlp_dim=mlp_dim,
            dropout=0.1
        )

        self.fuse_conv = nn.Identity()

    def forward(self, x):
        spatial_feat = x

        x_fp32 = x.to(torch.float32)

        f = fft.rfft2(x_fp32, dim=(-2, -1))
        f_filtered = f * torch.sigmoid(self.freq_weight.to(torch.float32))
        freq_feat = fft.irfft2(f_filtered, s=x_fp32.shape[-2:], dim=(-2, -1))  # [B, C, H, W]

        freq_feat = freq_feat.to(x.dtype)

        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        out = self.fuse_conv(fused[:, :self.c1] + fused[:, self.c1:])

        out = self.mobilevit(out)
        return out