__all__ = ["train_step", "test_step"]

import torch


def train_step(classifier, train_loader, loss, optimizer, device):
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(images)
        loss_val = loss(outputs, labels)
        loss_val.backward()
        optimizer.step()
        total_loss += loss_val.item()
    return total_loss / len(train_loader)

def test_step(classifier, test_loader, loss, device):
    total_loss = 0
    correct_num = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            loss_val = loss(outputs, labels)
            total_loss += loss_val.item()
            pred = outputs.argmax(1)
            correct_num += pred.eq(labels).sum().item()
    return total_loss / len(test_loader), correct_num