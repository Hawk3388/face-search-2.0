import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights


def inference_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        backbone = resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projection = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def encode(self, image):
        return F.normalize(self.projection(self.encoder(image)), p=2, dim=1)

    def forward(self, first, second):
        return F.cosine_similarity(self.encode(first), self.encode(second), dim=1)

    def train(self, mode=True):
        super().train(mode)
        # Freezing gradients alone does not freeze BatchNorm running statistics.
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
                    module.eval()
        return self
