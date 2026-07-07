from typing import Tuple

def dwt2d(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    B, C, H, W = x.shape
    pad_h = (2 - H % 2) % 2
    pad_w = (2 - W % 2) % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    x01 = x[..., ::2, :]
    x02 = x[..., 1::2, :]
    x1 = x01[..., ::2]
    x2 = x01[..., 1::2]
    x3 = x02[..., ::2]
    x4 = x02[..., 1::2]
    ll = (x1 + x2 + x3 + x4) * 0.5 + 1e-8
    lh = (x1 - x2 + x3 - x4) * 0.5 + 1e-8
    hl = (x1 + x2 - x3 - x4) * 0.5 + 1e-8
    hh = (x1 - x2 - x3 + x4) * 0.5 + 1e-8
    if pad_h:
        ll, lh, hl, hh = [t[..., :-1, :] for t in [ll, lh, hl, hh]]
    if pad_w:
        ll, lh, hl, hh = [t[..., :-1] for t in [ll, lh, hl, hh]]
    return hh, hl, lh, ll
def iwt2d(hh, hl, lh, ll):
    B, C, H, W = ll.shape
    up = torch.zeros((B, C, H*2, W*2), dtype=ll.dtype, device=ll.device)
    up[..., ::2, ::2] = (ll + hl + lh + hh) * 0.5
    up[..., 1::2, ::2] = (ll + hl - lh - hh) * 0.5
    up[..., ::2, 1::2] = (ll - hl + lh - hh) * 0.5
    up[..., 1::2, 1::2] = (ll - hl - lh + hh) * 0.5
    return up

class DWTConv(nn.Module):

    def __init__(self, c_in: int, c_out: int):
        super().__init__()

        self.hh_conv = nn.Conv2d(c_in, c_out//3, 1, 1, 0, bias=False)
        self.hl_conv = nn.Conv2d(c_in, c_out//3, 1, 1, 0, bias=False)
        self.lh_conv = nn.Conv2d(c_in, c_out - 2*(c_out//3), 1, 1, 0, bias=False)

        self.omega = nn.Parameter(torch.tensor([2.0, 1.5, 1.5]))  # hh,hl,lh2.0, 1.5, 1.5
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        hh, hl, lh, _ = dwt2d(x)
        hh_out = self.hh_conv(hh) * self.omega[0]
        hl_out = self.hl_conv(hl) * self.omega[1]
        lh_out = self.lh_conv(lh) * self.omega[2]
        out = torch.cat([hh_out, hl_out, lh_out], dim=1)
        return out

class IWTConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out * 4, 1, 1, 0, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x = self.conv(x)
        hh, hl, lh, ll = torch.chunk(x, 4, dim=1)
        return iwt2d(hh, hl, lh, ll) * self.scale

class CBL(nn.Module):
    def __init__(self, c_in: int, c_out: int, shortcut=False):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.shortcut = shortcut
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return x + self.act(self.bn(self.conv(x))) if self.shortcut else \
               self.act(self.bn(self.conv(x)))

class WEFPBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mid = max(1, c_out // 2)


        self.reduce = nn.Conv2d(c_in, c_in, 1, 1, 0, bias=False) if c_in != c_out else nn.Identity()


        self.dwt1   = DWTConv(c_in, c_out)
        self.dwt_h  = DWTConv(self.mid, max(1, self.mid // 2))
        self.dwt_l  = DWTConv(self.mid, max(1, self.mid // 2))
        self.iwt_h  = IWTConv(max(1, self.mid // 2), self.mid)
        self.iwt_l  = IWTConv(max(1, self.mid // 2), self.mid)

        self.cbl_h  = CBL(self.mid, self.mid, shortcut=True)
        self.cbl_l  = CBL(self.mid, self.mid, shortcut=True)

        self.out_conv = CBL(c_out, c_out)
        self.adjust_out = nn.Conv2d(c_out, c_out, 1, 1, 0) if c_out % 2 != 0 else nn.Identity()

    def forward(self, x):
        x = self.reduce(x)
        f1 = self.dwt1(x)


        split_size = self.mid
        f1_h = f1[:, :split_size]
        f1_l = f1[:, split_size:]


        if f1_h.shape[1] != self.mid:
            f1_h = F.adaptive_avg_pool2d(f1_h, (self.mid, 1))
        if f1_l.shape[1] != self.mid:
            f1_l = F.adaptive_avg_pool2d(f1_l, (self.mid, 1))

        f2_hh = self.dwt_h(f1_h)
        f2_ll = self.dwt_l(f1_l)

        f1_h_commu = self.iwt_h(f2_hh)
        f1_l_commu = self.iwt_l(f2_ll)


        f1_h_commu = F.interpolate(f1_h_commu, size=f1_h.shape[-2:], mode='bilinear', align_corners=True)
        f1_l_commu = F.interpolate(f1_l_commu, size=f1_l.shape[-2:], mode='bilinear', align_corners=True)


        attn_h = torch.sigmoid(F.avg_pool2d(f1_h_commu, kernel_size=5, stride=1, padding=2))
        attn_l = torch.sigmoid(F.avg_pool2d(f1_l_commu, kernel_size=5, stride=1, padding=2))

        f1_h_upd = self.cbl_h(f1_h + f1_h_commu * attn_h)
        f1_l_upd = self.cbl_l(f1_l + f1_l_commu * attn_l)

        out = torch.cat([f1_h_upd, f1_l_upd], dim=1)
        out = self.out_conv(out)
        return self.adjust_out(out)