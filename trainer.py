import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from util import get_classification_report, plot_confusion_matrix, plot_loss_curve


class Trainer():
    def __init__(self, model, train_loader, test_loader, lr=4e-4, epochs=50):
        # using cuda is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criteria = nn.CrossEntropyLoss()
        # AdamW = Adam + Weight Decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        # change the learning rate amid learning to get better performance
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, verbose=True)
        
    def predict(self, state):
        self.model.eval()
        true = []
        pred = []
        loader = self.train_loader if state == 'train' else self.test_loader
        for data, target in loader:
            data = data.to(self.device)
            with torch.no_grad():
                output = self.model(data)
            _, predicted = torch.max(output.data, dim=1)
            true.extend(target.tolist())
            pred.extend(predicted.tolist())

        return true, pred

    def train(self, epochs=50):
        loss_list = []
        for epoch in range(epochs):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criteria(output, target)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            loss_list.append(loss.item())
            true, pred = self.predict('test')
            accuracy = accuracy_score(true, pred) * 100
            print('Ep: {}/{}, Loss: {:.6f}, Accuracy: {:.2f}%'.format(epoch+1, epochs, loss.item(), accuracy))
        plot_loss_curve(loss_list)
        torch.save(self.model.state_dict(), 'vision_transformer.pth')

    def test(self):
        for state in 'train', 'test':
            true, pred = self.predict(state)
            accuracy = accuracy_score(true, pred) * 100
            cm = confusion_matrix(true, pred)
            plot_confusion_matrix(cm, state)
            print('Accuracy for {}: {:.2f}%'.format(state, accuracy))

            if state == 'test':
                get_classification_report(true, pred, state)
