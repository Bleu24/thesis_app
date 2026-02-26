import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint

# =================================================================
# 1. U-NET (Exact Match to train_unet.py)
# =================================================================

class DoubleConv(nn.Module):
    _VALID_NORMS = {"batch", "instance", "group", "none"}

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        if norm not in self._VALID_NORMS:
            raise ValueError(f"Unsupported norm '{norm}'")
        self.norm = norm
        use_bias = norm == "none"
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            self._make_norm(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            self._make_norm(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def _make_norm(self, channels: int) -> nn.Module:
        if self.norm == "batch":
            return nn.BatchNorm2d(channels)
        if self.norm == "instance":
            return nn.InstanceNorm2d(channels, affine=True)
        if self.norm == "group":
            # Exact logic from your train_unet.py
            groups = min(8, channels)
            while channels % groups != 0 and groups > 1:
                groups -= 1
            return nn.GroupNorm(groups, channels)
        return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 1, base_channels: int = 64, 
                 max_channels: int = 512, depth: int = 5, norm: str = "group", use_checkpoint: bool = False) -> None:
        super().__init__()
        
        n_channels, n_classes = in_channels, out_channels
        
        if depth < 2: raise ValueError("UNet depth must be >= 2")
        self.use_checkpoint = use_checkpoint
        self.pool = nn.MaxPool2d(2)

        enc_blocks = []
        skip_channels = []
        in_ch = n_channels
        out_ch = min(max_channels, base_channels)
        
        for _ in range(depth):
            enc_blocks.append(DoubleConv(in_ch, out_ch, norm=norm))
            skip_channels.append(out_ch)
            in_ch = out_ch
            out_ch = min(max_channels, out_ch * 2)

        bottleneck_out = min(max_channels, in_ch * 2)
        self.bottleneck = DoubleConv(in_ch, bottleneck_out, norm=norm)

        up_blocks = []
        dec_blocks = []
        prev_ch = bottleneck_out
        for skip_ch in reversed(skip_channels):
            up_blocks.append(nn.ConvTranspose2d(prev_ch, skip_ch, kernel_size=2, stride=2))
            dec_blocks.append(DoubleConv(skip_ch * 2, skip_ch, norm=norm))
            prev_ch = skip_ch

        self.encoders = nn.ModuleList(enc_blocks)
        self.upconvs = nn.ModuleList(up_blocks)
        self.decoders = nn.ModuleList(dec_blocks)
        self.out_conv = nn.Conv2d(prev_ch, n_classes, kernel_size=1)

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(block, x)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = self._run_block(encoder, x)
            skips.append(x)
            x = self.pool(x)

        x = self._run_block(self.bottleneck, x)

        for upsample, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.size(-2) - x.size(-2)
                diff_x = skip.size(-1) - x.size(-1)
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([skip, x], dim=1)
            x = self._run_block(decoder, x)

        return self.out_conv(x)

# =================================================================
# 2. RESNET (Already fixed, keeping it same)
# =================================================================

def _inflate_first_conv(model: nn.Module, in_channels: int) -> None:
    old_conv = model.conv1
    new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                         padding=old_conv.padding, bias=False)
    with torch.no_grad():
        if old_conv.weight.shape[1] == 3 and in_channels >= 3:
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if in_channels > 3:
                mean_extra = old_conv.weight.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_conv.weight[:, c:c+1, :, :] = mean_extra
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.conv1 = new_conv

class AuxMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class ResNetRegression(nn.Module):
    def __init__(self, in_channels: int = 7, aux_features: int = 8, backbone: str = "resnet18", 
                 pretrained: bool = False, fusion_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.aux_features_dim = aux_features
        if backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        else: raise ValueError("Unsupported backbone")
        _inflate_first_conv(model, in_channels)
        self.backbone = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.img_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.aux_head = AuxMLP(in_features=aux_features, hidden=fusion_hidden, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2*fusion_hidden, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, img: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        if aux is None: aux = torch.zeros(img.size(0), self.aux_features_dim).to(img.device)
        x = self.backbone(img)
        x = self.avgpool(x)
        x = self.img_head(x)
        a = self.aux_head(aux)
        z = torch.cat([x, a], dim=1)
        y = self.fusion(z).squeeze(1)
        return y