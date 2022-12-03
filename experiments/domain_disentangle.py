import torch
from models.base_model import DomainDisentangleModel

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion_1 = torch.nn.CrossEntropyLoss()
        self.criterion_2 = torch.nn.MSELoss()

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data):
        x, y, domain = data
        x = x.to(self.device)
        y = y.to(self.device)
        domain_type = domain
        domain = domain.to(self.device)

        results = self.model(x, domain)
        if domain_type == 0:
            loss_1 = self.criterion_1(results[2], y)
            loss_2 = -self.criterion_1(results[3], domain) 
            loss_3 = self.criterion_2(results[0], results[1])
            self.optimizer.zero_grad()
            loss_1.backward(retain_graph=True)
            loss_2.backward(retain_graph=True)
            loss_3.backward(retain_graph=True)
            self.optimizer.step()
            loss = loss_1+loss_2+loss_3
            return loss.item()
        else:
            loss_1 = self.criterion_1(results[2], domain)
            loss_2 = self.criterion_2(results[0], results[1])
            loss_1.backward(retain_graph=True)
            loss_2.backward(retain_graph=True)
            self.optimizer.step()
            loss = loss_1+loss_2
            return loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                _, _, category_class_cclf, _ , _ , _  = self.model(x)
                loss += self.criterion_1(category_class_cclf, y)
                pred = torch.argmax(category_class_cclf, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss