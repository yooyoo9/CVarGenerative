import numpy as np
import torch

def train(model, optimizer, criterion, train_loader,
          valid_loader, epochs, device, path):
    model.train()
    for epoch_idx in range(epochs):
        print(f"Epoch {epoch_idx+1} of {epochs}")
        running_loss = 0.0
        for batch_idx, (data, *_) in enumerate(train_loader):
            data = data.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            recons, mu, logvar = model(data)
            loss = model.loss(data, recons, mu, logvar, criterion)
            loss = loss.sum()
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(train_loader.dataset)
        val_loss = evaluate(model, criterion, valid_loader, device, epoch_idx)
        print(f"Train Loss: {train_loss:.4f}   Val Loss: {val_loss:.4f}")
    torch.save(model, path)


def evaluate(model, criterion, data_loader, device, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (data, *_) in data_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            recons, mu, logvar = model(data)
            loss = model.loss(data, recons, mu, logvar, criterion)
            loss = loss.sum()
            running_loss += loss.item()
    val_loss = running_loss/len(data_loader.dataset)
    return val_loss
