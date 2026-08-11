from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


SUPPORTED_MODELS = ("vit_small", "vit_large", "efficientnetv2_l")


class MultiLabelClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_name: str,
        dropout_rate: float = 0.7,
        stochastic_depth_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name

        if model_name == "vit_small":
            self.backbone = timm.create_model(
                "vit_small_patch16_224",
                pretrained=True,
                drop_path_rate=stochastic_depth_prob,
            )
            self.feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif model_name == "vit_large":
            self.backbone = timm.create_model(
                "vit_large_patch16_224",
                pretrained=True,
                drop_path_rate=stochastic_depth_prob,
            )
            self.feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif model_name == "efficientnetv2_l":
            self.backbone = timm.create_model(
                "tf_efficientnetv2_l.in21k_ft_in1k",
                pretrained=True,
                drop_path_rate=stochastic_depth_prob,
            )
            self.feature_dim = self.backbone.num_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported model {model_name!r}. Choose from {SUPPORTED_MODELS}."
            )

        self.projection = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.LayerNorm(self.feature_dim),
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.projection(features)
        return self.classifier(features)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        return (((1 - pt) ** self.gamma) * bce_loss).mean()
