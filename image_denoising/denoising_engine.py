__all__ = ["train_step", "test_step"]

import torch


def train_step(denoiser, train_loader, loss, optimizer, device):
    denoiser.train()
    total_loss = 0.0
    for train_imgs, target_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        target_imgs = target_imgs.to(device)
        outputs = denoiser(train_imgs)
        loss_value = loss(outputs, target_imgs)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value
    return total_loss / len(train_loader)

def test_step(denoiser, test_loader, loss, device):
    denoiser.eval()
    total_loss = 0.0
    with torch.no_grad():
        for test_imgs, target_imgs in test_loader:
            test_imgs = test_imgs.to(device)
            target_imgs = target_imgs.to(device)
            outputs = denoiser(test_imgs)
            loss_value = loss(outputs, target_imgs)
            total_loss += loss_value
    return total_loss / len(test_loader)