from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets

import train_zoo


def test_transformed_subset_applies_transform(tiny_image_dataset):
    base_dataset = datasets.ImageFolder(root=str(tiny_image_dataset))
    subset = Subset(base_dataset, [0])

    transformed_subset = train_zoo.TransformedSubset(
        subset,
        transform=lambda image: image.resize((16, 16)),
    )

    image, label = transformed_subset[0]

    assert isinstance(image, Image.Image)
    assert image.size == (16, 16)
    assert isinstance(label, int)


def test_main_returns_if_dataset_missing(monkeypatch, capsys):
    monkeypatch.setattr(train_zoo, "DATA_DIR", "missing-dataset-dir")

    train_zoo.main()

    stdout = capsys.readouterr().out
    assert "Dataset directory 'missing-dataset-dir' not found" in stdout
