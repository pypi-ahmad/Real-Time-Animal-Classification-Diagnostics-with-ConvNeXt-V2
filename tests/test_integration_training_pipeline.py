from pathlib import Path

import torch
import torch.nn as nn

import train_zoo


class TinyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, num_classes),
        )

    def forward(self, inputs):
        return self.net(inputs)


def test_training_pipeline_creates_valid_bundle(tmp_path, tiny_image_dataset, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_zoo, "DATA_DIR", str(tiny_image_dataset))
    monkeypatch.setattr(train_zoo, "EPOCHS", 1)
    monkeypatch.setattr(train_zoo, "PATIENCE", 1)
    monkeypatch.setattr(train_zoo, "BATCH_SIZE", 2)
    monkeypatch.setattr(train_zoo, "NUM_CLASSES", 2)

    def fake_create_model(_name, pretrained, num_classes):
        assert pretrained is True
        return TinyClassifier(num_classes=num_classes)

    monkeypatch.setattr(train_zoo.timm, "create_model", fake_create_model)

    train_zoo.main()

    bundle_path = Path("zoo_bundle.pth")
    assert bundle_path.exists()

    artifact = torch.load(bundle_path, map_location="cpu", weights_only=True)

    assert set(artifact.keys()) == {"model_state", "class_names", "metrics"}
    assert artifact["class_names"] == ["Dog", "Cat"]
    assert "accuracy" in artifact["metrics"]
    assert "f1_score" in artifact["metrics"]
