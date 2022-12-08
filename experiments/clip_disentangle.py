import torch
import clip
from models.base_model import DomainDisentangleModel

class CLIPDisentangleExperiment:
    
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

    def clip_encoder(self):
        # Load CLIP model and freeze it
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
        clip_model.to(device)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False

        # To use the textual encoder:
        description = 'a picture of a small dog playing with a ball'
        tokenized_text = clip.tokenize(description).to(device)

        return clip_model.encode_text(tokenized_text)


    def train_iteration(self, data, label):
        x, y, domain = data
        x = x.to(self.device)
        y = y.to(self.device)
        domain = domain.to(self.device)

        results = self.model(x, label, experiment='clip')
        clip = self.clip_encoder()
        if label == 0:
            # Training with source
            loss_1 = self.criterion_1(results[2], y)
            loss_2 = self.criterion_1(results[3], domain)
            loss_3 = -self.criterion_1(results[4], y)
            loss_4 = -self.criterion_1(results[5], domain)
            loss_5 = self.criterion_2(clip, results[6])
            loss_6 = self.criterion_2(results[0], results[1])
            self.optimizer.zero_grad()
            loss_1.backward(retain_graph=True)
            loss_2.backward(retain_graph=True)
            loss_3.backward(retain_graph=True)
            loss_4.backward(retain_graph=True)
            loss_5.backward(retain_graph=True)
            loss_6.backward(retain_graph=True)
            self.optimizer.step()
            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
            return loss.item()
        else:
            # Training with target
            loss_1 = self.criterion_1(results[2], domain) 
            loss_2 = -self.criterion_1(results[3], domain)
            loss_3 = self.criterion_2(results[4], clip)
            loss_4 = self.criterion_2(results[0], results[1])
            self.optimizer.zero_grad()
            loss_1.backward(retain_graph=True)
            loss_2.backward(retain_graph=True)
            loss_3.backward(retain_graph=True)
            loss_4.backward(retain_graph=True)
            self.optimizer.step()
            loss = loss_1 + loss_2 + loss_3 + loss_4
            return loss.item()

    def validate(self, loader, label):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y, domain in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                domain = domain.to(self.device)

                if label == 0:
                    # Validation with source                
                    _, _, category_class, _, _, _, _ = self.model(x, label, experiment='clip')
                    loss += self.criterion_1(category_class, y)
                    pred = torch.argmax(category_class, dim=-1)
                    accuracy += (pred == y).sum().item()
                    count += x.size(0)
                elif label == 1:
                    # Validation with target
                    _, _, domain_class, _, _ = self.model(x, label, experiment='clip')
                    loss += self.criterion_1(domain_class, domain)
                    pred = torch.argmax(domain_class, dim=-1)
                    accuracy += (pred == domain).sum().item()
                    count += x.size(0)
                else:
                    # Testing
                    result = self.model(x, label, experiment='clip')
                    loss += self.criterion_1(result, y)
                    pred = torch.argmax(result, dim=-1)
                    accuracy += (pred==y).sum().item()
                    count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss