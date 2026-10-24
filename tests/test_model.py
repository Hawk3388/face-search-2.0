import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from model2.network import SiameseNetwork
from model2.train import ContrastiveLoss


def test_frozen_batchnorm_and_normalized_embeddings():
    model = SiameseNetwork(embedding_dim=16, pretrained=False)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    model.train()
    assert not model.encoder.bn1.training
    model.eval()
    with torch.inference_mode():
        embeddings = model.encode(torch.zeros(2, 3, 64, 64))
    assert embeddings.shape == (2, 16)
    torch.testing.assert_close(torch.linalg.norm(embeddings, dim=1), torch.ones(2))


def test_contrastive_loss_encourages_correct_pair_order():
    criterion = ContrastiveLoss()
    labels = torch.tensor([1.0, 0.0])
    assert criterion(torch.tensor([1.0, 0.0]), labels) < criterion(torch.tensor([0.0, 1.0]), labels)
