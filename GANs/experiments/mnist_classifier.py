import os
import torch
from torchvision import datasets, transforms
from torch import nn


class MnistClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = "../models/mnist_usual/mnist_classifier"
        train = False
        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path, map_location=torch.device("cpu"))
        else:
            train = True
            input_size = 784
            hidden_sizes = [625]
            output_size = 10
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], output_size),
                nn.Softmax(dim=1),
            )
        self.model.to(self.device)

        train_set = datasets.MNIST(
            root="../../input/mnist",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        valid_set = datasets.MNIST(
            root="../../input/mnist",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=500, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=500, shuffle=True
        )
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.9
        )
        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()

        if train:
            self.train(100)
            self.evaluate()

    def train(self, epochs):
        self.model.train()
        for epoch_idx in range(epochs):
            running_loss = 0.0
            for data, labels in self.train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                data = data.view(data.shape[0], -1)
                self.optimizer.zero_grad()
                log_prob = self.model(data)
                loss = self.criterion(log_prob, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(
                "Epoch {} - Training loss: {}".format(
                    epoch_idx, running_loss / len(self.train_loader)
                )
            )
        torch.save(self.model, self.model_path)

    def evaluate(self):
        self.model.eval()
        correct_count, all_count = 0, 0
        for data, true_labels in self.val_loader:
            data = data.to(self.device)
            true_labels = true_labels.to(self.device)
            data = data.view(data.shape[0], -1)
            log_prob = self.model(data)
            pred_labels = torch.argmax(log_prob, dim=1)
            correct_count += torch.sum(pred_labels == true_labels)
            all_count += len(true_labels)

        print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy =", (correct_count / all_count))

    def predict(self, data):
        data = data.to(self.device)
        data = data.view(data.shape[0], -1)
        log_prob = self.model(data)
        pred_labels = torch.argmax(log_prob, dim=1).cpu().numpy()
        return pred_labels
