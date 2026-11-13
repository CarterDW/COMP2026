"""Networks and the training loop (notebook 10's loop, on the GPU)."""
import torch

DEVICE = "cuda"
torch.backends.cudnn.deterministic = True    # same seed, same numbers
torch.backends.cudnn.benchmark = False


def cnn(in_ch=3, n_out=2, size=120):
    """Notebook 10's CNN. The last Linear's input comes from the size formula:
    conv5 (no padding): H -> H - 4;  MaxPool2d(2): H -> floor(H / 2)."""
    h = ((size - 4) // 2 - 4) // 2          # 120 -> 116 -> 58 -> 54 -> 27
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_ch, 8, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 16, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(), torch.nn.Linear(16 * h * h, n_out))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def accuracy(model, x, y, batch=1000):
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x[i:i + batch].to(DEVICE)).argmax(1).cpu()
                          for i in range(0, len(x), batch)])
    model.train()
    return (pred == y).float().mean().item()


def train(model, x_train, y_train, x_test, y_test, n_epochs=15, batch_size=64, lr=1e-3, seed=0):
    """Shuffled minibatches with Adam; returns test accuracy after every epoch."""
    torch.manual_seed(seed)
    model.to(DEVICE)
    xt, yt = x_train.to(DEVICE), y_train.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(n_epochs):
        order = torch.randperm(len(xt), device=DEVICE)
        for start in range(0, len(xt), batch_size):
            b = order[start:start + batch_size]
            optimizer.zero_grad()
            criterion(model(xt[b]), yt[b]).backward()
            optimizer.step()
        history.append(accuracy(model, x_test, y_test))
        print(f"epoch {epoch + 1:2d}: test accuracy {100 * history[-1]:.1f}%")
    return history
