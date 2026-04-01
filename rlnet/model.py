from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class RLNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.temporal_model = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        temporal_dim = hidden_size * (2 if bidirectional else 1)
        self.temporal_classifier = nn.Linear(temporal_dim, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, 1),
        )

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = frames.shape
        flattened = frames.reshape(batch_size * sequence_length, channels, height, width)
        features = self.backbone(flattened)
        temporal_input = features.reshape(batch_size, sequence_length, -1)
        sequence_output, _ = self.temporal_model(temporal_input)
        return sequence_output

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        sequence_output = self.encode_sequence(frames)
        pooled = torch.cat([sequence_output.mean(dim=1), sequence_output.amax(dim=1)], dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def temporal_logits(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_output = self.encode_sequence(frames)
        batch_size, sequence_length, hidden_size = sequence_output.shape
        timewise = self.temporal_classifier(sequence_output.reshape(batch_size * sequence_length, hidden_size))
        timewise = timewise.reshape(batch_size, sequence_length).contiguous()
        pooled = torch.cat([sequence_output.mean(dim=1), sequence_output.amax(dim=1)], dim=1)
        final_logits = self.classifier(pooled).squeeze(-1)
        return final_logits, timewise

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
