class ClassBalancedLoss(nn.Module):

    def __init__(self, num_classes=6, class_instance_counts=None, beta=0.75, gamma=0.35, bfnl_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        if class_instance_counts is None:
            class_instance_counts = [100] * num_classes
        assert len(class_instance_counts) == num_classes, \
            f"len{len(class_instance_counts)}num{num_classes}not"
        self.register_buffer(
            'SAMPLES_PER_CLASS',
            torch.tensor(class_instance_counts, dtype=torch.float32)
        )
        def _logit(p):
            p = torch.as_tensor(p).clamp(1e-7, 1 - 1e-7)
            return (p / (1 - p)).log()

        self._beta_raw  = nn.Parameter(_logit((beta - 0.5) / 0.45))
        self._gamma_raw = nn.Parameter(_logit((gamma - 0.1) / 0.4))
        self._bfnl_raw  = nn.Parameter(_logit(bfnl_weight / 5.))
        self.register_buffer('base_weights', None)
        self.register_buffer('class_weights', None)
        self.register_buffer('_ap_ema', torch.zeros(num_classes))
        self.register_buffer('_last_epoch', torch.tensor(-1))
        self._init_weights()
        self._thresh_fp_raw = nn.Parameter(torch.logit(torch.tensor(0.3)))
        self._thresh_fn_raw = nn.Parameter(torch.logit(torch.tensor(0.5)))
        self.l2_lambda = 1e-4

    # ------------------------------------------------------------------
#     @property
#     def thresh_fp(self):
#         return float(torch.sigmoid(self._thresh_fp_raw))

#     @property
#     def thresh_fn(self):
#         return float(torch.sigmoid(self._thresh_fn_raw))

    @property
    def thresh_fp(self):
        return torch.sigmoid(self._thresh_fp_raw)   # 保持 Tensor

    @property
    def thresh_fn(self):
        return torch.sigmoid(self._thresh_fn_raw)


    def _init_weights(self):
        base = 1.0 / (self.SAMPLES_PER_CLASS + 1e-5)
        base = base / base.max() * 2.0
        self.register_buffer('base_weights', base.clone())
        self.register_buffer('class_weights', base.clone())
#     @property
#     def beta(self):
#         return float(torch.sigmoid(self._beta_raw) * 0.45 + 0.5)

#     @property
#     def gamma(self):
#         return float(torch.sigmoid(self._gamma_raw) * 0.4 + 0.1)

#     @property
#     def bfnl_weight(self):
#         return float(torch.sigmoid(self._bfnl_raw) * 5.)


    @property
    def beta(self):
        return torch.sigmoid(self._beta_raw) * 0.45 + 0.5
    @property
    def gamma(self):
        return torch.sigmoid(self._gamma_raw) * 0.4 + 0.1

    @property
    def bfnl_weight(self):
        return torch.sigmoid(self._bfnl_raw) * 5.0
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_weights(self, ap50: np.ndarray, epoch: int,
                       mutual_fp: float = 0.0, mutual_fn: float = 0.0,
                       ema_momentum: float = None):

#         if epoch == self._last_epoch.item():
#             return

        if (epoch % 10 != 0) and (epoch == self._last_epoch.item()):
            return
        
        self._last_epoch.copy_(torch.tensor(epoch))
        device = self.class_weights.device
        ap = torch.as_tensor(ap50, dtype=torch.float32, device=device).clamp(0.01, 1.0)
        if ema_momentum is None:
            ema_momentum = max(0.05, 0.5 - epoch * 0.01)
        self._ap_ema.lerp_(ap, ema_momentum)

        difficulty = 1.0 - self._ap_ema

        hardest_cls = int(torch.argmin(self._ap_ema))
        difficulty[hardest_cls] *= 1.0 + 0.3 * (mutual_fp + mutual_fn)
        difficulty[hardest_cls] = difficulty[hardest_cls].clamp_max(1.5)

        lt_factor = self.base_weights / self.base_weights.max()
        combined = difficulty * (1.0 + 0.5 * lt_factor)

        combined = combined / combined.sum() * self.num_classes
        
        combined = combined.clamp(0.4, 2.5)
        self.class_weights.copy_(combined)

    # ------------------------------------------------------------------
    def forward(self, logits, target):
        device = logits.device

        beta  = self.beta.to(device)
        gamma = self.gamma.to(device)
        bfnl_weight = self.bfnl_weight.to(device)
        thresh_fp = self.thresh_fp.to(device)
        thresh_fn = self.thresh_fn.to(device)

        B, A, C = logits.shape
        w = self.class_weights.view(1, 1, C).to(device)

        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        prob = torch.sigmoid(logits)

        focal = (target * (1 - prob) + (1 - target) * prob + 1e-8) ** gamma
        base_loss = (bce * focal * w).sum()

        if C > 2:
            hardest_cls = int(torch.argmin(self._ap_ema.to(device)))
            bg_mask = (target.sum(dim=2, keepdim=True) == 0)
            pred_hardest = prob[:, :, hardest_cls:hardest_cls+1]
            fp_mask = bg_mask & (pred_hardest > self.thresh_fp)
            fn_mask = (target[:, :, hardest_cls] == 1) & (pred_hardest.squeeze(-1) < self.thresh_fn)

            fp_count = fp_mask.sum().clamp_min(1)
            fn_count = fn_mask.sum().clamp_min(1)
            bfnl = (pred_hardest[fp_mask] ** 2).sum() / fp_count + \
                   ((1 - pred_hardest.squeeze(-1)[fn_mask]) ** 2).sum() / fn_count
        else:
            bfnl = torch.tensor(0., device=device)

        bfnl = bfnl * (bfnl_weight > 0).float() * bfnl_weight
        bfnl = bfnl * bfnl_weight.clamp_min(0.)

        l2_penalty = self.l2_lambda * (
            self._beta_raw ** 2 +
            self._gamma_raw ** 2 +
            self._bfnl_raw ** 2 +
            self._thresh_fp_raw ** 2 +
            self._thresh_fn_raw ** 2
        )
        return (base_loss + bfnl + l2_penalty) * 10

