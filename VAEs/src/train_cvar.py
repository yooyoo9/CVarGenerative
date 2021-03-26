import numpy as np
import torch

def cvar_train(model, optimizer, criterion, cvar, train_loader, adaptive_algorithm,
               valid_loader, epochs, alpha, device, path, save):
    for epoch_idx in range(epochs):
        print(f"Epoch {epoch_idx+1} of {epochs}")
        model.train()
        running_loss = 0.0
        for batch_idx, (data, idx) in enumerate(train_loader):
            data = data.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            cvar.zero_grad()

            recons, mu, logvar = model(data)
            loss = model.loss(data, recons, mu, logvar, criterion)

            weights = 1.0
            prob = adaptive_algorithm.probabilities
            adaptive_algorithm.update(
                1 - np.clip(loss.cpu().detach().numpy(), 0, 1), idx, prob
            )

            cvar_loss = (torch.tensor(weights).float().to(device) * cvar(loss)).mean()
            running_loss += loss.sum().item()
            loss = loss.mean()
            cvar_loss.backward()

            optimizer.step()
            cvar.step()

            adaptive_algorithm.normalize()

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = cvar_evaluate(model, criterion, valid_loader, alpha, device,
                                 epoch_idx)
        print(f"Train Loss: {train_loss:.4f}   Val Loss: {val_loss:.4f}")
    if save:
        torch.save(model, path)


def cvar_evaluate(model, criterion, data_loader, alpha, device, epoch):
    model.eval()
    k = int(np.ceil(alpha * len(data_loader.dataset)))
    top_k = None
    count = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, _ in data_loader:
            count += data.shape[0]
            data = data.view(data.size(0), -1)
            recons, mu, logvar = model(data)
            losses = model.loss(data, recons, mu, logvar, criterion).sort(descending=True)[0]
            if top_k is None:
                top_k = losses[:k]
            else:
                top_k = (torch.cat((top_k, losses))).sort(descending=True)[0]
                top_k = top_k[:k]
            running_loss += losses.sum().item()
    val_loss = running_loss/len(data_loader.dataset)
    return val_loss

