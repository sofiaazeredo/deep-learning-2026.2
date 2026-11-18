"""Modelos e checkpoints portáteis. Sem dependência de dados nem de internet (pretrained=False)."""
import torch
import torch.nn as nn
import torchvision.models as tvm

ARCHS = ("resnet18", "mobilenet_v3_large", "efficientnet_b0", "smallcnn")
CKPT_VERSION = 1


class SmallCNN(nn.Module):
    """CNN pequena do zero (Parte 0a). Larguras e dropout vêm da config."""

    def __init__(self, n_classes, widths=(32, 64, 128, 256, 256), dropout=0.3):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):
        return self.fc(self.drop(self.features(x).mean((2, 3))))


def build_model(arch, num_classes, pretrained=False, widths=None, dropout=0.3):
    """pretrained=True baixa pesos ImageNet (classificação supervisionada) pelo torchvision: precisa de internet."""
    if arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "smallcnn":
        if pretrained:
            raise ValueError("smallcnn é treinada do zero")
        m = SmallCNN(num_classes, tuple(widths) if widths else (32, 64, 128, 256, 256), dropout)
    else:
        raise ValueError(f"arch desconhecida: {arch} (use {ARCHS})")
    return m


def count_params(model):
    """Contagem exigida pelo enunciado."""
    return sum(p.numel() for p in model.parameters())


def target_layer(model, arch):
    """Última camada convolucional, para o Grad-CAM."""
    if arch == "resnet18":
        return model.layer4[-1]
    if arch in ("mobilenet_v3_large", "efficientnet_b0"):
        return model.features[-1]
    if arch == "smallcnn":
        return model.features[-2]
    raise ValueError(arch)


def save_checkpoint(path, model, arch, class_names, preprocess, meta=None, arch_kwargs=None):
    """Checkpoint autocontido: arquitetura, nomes de classes (de classes.csv) e pré-processamento.
    O predict.py só precisa deste arquivo."""
    torch.save(dict(version=CKPT_VERSION, arch=arch, arch_kwargs=arch_kwargs or {},
                    num_classes=len(class_names), class_names=list(class_names), preprocess=dict(preprocess),
                    meta=meta or {}, num_params=count_params(model),
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ck["arch"], ck["num_classes"], pretrained=False, **ck.get("arch_kwargs", {}))
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.eval(), ck
