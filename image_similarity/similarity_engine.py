__all__ = ["train_step", "test_step", "create_embeading"]

import torch


def train_step(encoder, decoder, train_loader, loss, optimizer, device):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    for train_imgs, target_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        target_imgs = target_imgs.to(device)
        en_outputs = encoder(train_imgs)
        outputs = decoder(en_outputs)
        loss_value = loss(outputs, target_imgs)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value
    return total_loss / len(train_loader)

def test_step(encoder, decoder, test_loader, loss, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for test_imgs, target_imgs in test_loader:
            test_imgs = test_imgs.to(device)
            target_imgs = target_imgs.to(device)
            en_outputs = encoder(test_imgs)
            outputs = decoder(en_outputs)
            loss_value = loss(outputs, target_imgs)
            total_loss += loss_value
    return total_loss / len(test_loader)

def create_embeading(encoder, full_loader, device):
    encoder.eval()
    embeadings = torch.empty(0)
    with torch.no_grad():
        for train_img, target_imgs in full_loader:
            train_img = train_img.to(device)
            encoded_imgs = encoder(train_img).cpu()
            embeadings = torch.cat((embeadings, encoded_imgs), dim=0)
    return embeadings