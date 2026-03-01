import torch
import torch.nn as nn

import app


class TinyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, num_classes),
        )

    def forward(self, inputs):
        return self.net(inputs)


def test_load_test_data_reconstructs_expected_split(tiny_image_dataset, monkeypatch):
    app.load_test_data.clear()
    monkeypatch.setattr(app, "DATA_DIR", str(tiny_image_dataset))

    test_loader, classes = app.load_test_data()

    assert test_loader is not None
    assert classes == ["cane", "gatto"]
    assert len(test_loader.dataset) == 2


def test_load_zoo_model_success(monkeypatch):
    app.load_zoo_model.clear()

    model_stub = TinyClassifier(num_classes=10)

    def fake_create_model(_name, pretrained, num_classes):
        assert pretrained is False
        assert num_classes == 10
        return model_stub

    checkpoint = {
        "model_state": model_stub.state_dict(),
        "class_names": [f"class_{index}" for index in range(10)],
    }

    monkeypatch.setattr(app.timm, "create_model", fake_create_model)
    monkeypatch.setattr(app.torch, "load", lambda *_args, **_kwargs: checkpoint)

    model, class_names = app.load_zoo_model()

    assert model is model_stub
    assert model.training is False
    assert isinstance(class_names, list)
    assert len(class_names) == 10

    sample = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
    with torch.no_grad():
        output = model(sample)
    assert tuple(output.shape) == (1, 10)
