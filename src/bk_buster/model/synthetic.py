import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        torch.manual_seed(42)
        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        return x
    
def mlp_train(model, train_loader, optimizer, criterion, device):
    total_loss = 0
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        x = batch.x.float()
        y = batch.y.long().view(-1)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def mlp_test(model, test_loader, device):
    total = 0
    total_correct = 0
    model.eval()
    for batch in test_loader:
        batch = batch.to(device)
        x = batch.x.float()
        y = batch.y.long().view(-1)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += len(y)
    return total_correct / total

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def gcn_train(model, train_loader, optimizer, criterion, device):
    total_loss = 0
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        labels = batch.y.long().view(-1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def gcn_test(model, test_loader, device):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            labels = batch.y.long().view(-1)
            total_correct += (pred == labels).sum().item()
            total += labels.size(0)
            # print(f"batch correct: {total_correct}, batch total: {total}")
        accuracy = total_correct / total
    return accuracy

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
def gat_train(model, train_loader, optimizer, criterion, device):
    total_loss = 0
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        labels = batch.y.long().view(-1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def gat_test(model, test_loader, device):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            labels = batch.y.long().view(-1)
            total_correct += (pred == labels).sum().item()
            total += labels.size(0)
        accuracy = total_correct / total
    return accuracy