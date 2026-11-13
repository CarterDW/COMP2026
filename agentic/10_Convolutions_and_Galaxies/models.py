"""Networks and the training loop (notebook 10's loop, on the GPU)."""
import torch

DEVICE = "cuda"
torch.backends.cudnn.deterministic = True    # same seed, same numbers
torch.backends.cudnn.benchmark = False


def cnn(in_ch=3, n_out=2, size=120, blocks=3):
    """Conv(5x5) -> ReLU -> MaxPool2d(2) blocks with 8, 16, 32 channels, then one Linear.

    blocks=2 is notebook 10's CNN. The last Linear's input comes from the size formula:
    conv5 (no padding): H -> H - 4;  MaxPool2d(2): H -> floor(H / 2).
    At 120: 120 -> 116 -> 58 -> 54 -> 27 -> 23 -> 11.
    """
    channels = [in_ch, 8, 16, 32][:blocks + 1]
    layers, h = [], size
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [torch.nn.Conv2d(c_in, c_out, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2)]
        h = (h - 4) // 2
    return torch.nn.Sequential(*layers, torch.nn.Flatten(),
                               torch.nn.Linear(channels[-1] * h * h, n_out))


class TwoHead(torch.nn.Module):
    """A shared conv trunk (the cnn() blocks, flattened) with two 2-way heads."""

    def __init__(self, in_ch=3, size=120, blocks=3):
        super().__init__()
        full = cnn(in_ch=in_ch, size=size, blocks=blocks)
        self.trunk = torch.nn.Sequential(*list(full)[:-1])           # everything but the Linear
        n_feat = full[-1].in_features
        self.morph = torch.nn.Linear(n_feat, 2)
        self.color = torch.nn.Linear(n_feat, 2)

    def forward(self, x):
        h = self.trunk(x)
        return self.morph(h), self.color(h)


def train_two_head(model, x, y_morph, y_color, n_epochs=15, batch_size=64, lr=1e-3, seed=0):
    """Both heads trained together: loss = CE(morphology) + CE(color)."""
    torch.manual_seed(seed)
    model.to(DEVICE)
    x, ym, yc = x.to(DEVICE), y_morph.to(DEVICE), y_color.to(DEVICE)
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        order = torch.randperm(len(x), device=DEVICE)
        for s in range(0, len(x), batch_size):
            b = order[s:s + batch_size]
            opt.zero_grad()
            pm, pc = model(x[b])
            (ce(pm, ym[b]) + ce(pc, yc[b])).backward()
            opt.step()
    return model


def predict_two_head(model, x, batch=500, d4=False):
    """Softmax P(class 1) for both heads. d4=True averages the logits over the 8
    rotations and reflections of the square, which makes the output exactly D4-invariant."""
    model.eval()
    outs_m, outs_c = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(DEVICE)
            views = [xb] if not d4 else [torch.rot90(v, k, dims=(2, 3)) for v in (xb, xb.flip(3))
                                         for k in range(4)]
            lm = torch.stack([model(v)[0] for v in views]).mean(0)
            lc = torch.stack([model(v)[1] for v in views]).mean(0)
            outs_m.append(torch.softmax(lm, 1)[:, 1].cpu())
            outs_c.append(torch.softmax(lc, 1)[:, 1].cpu())
    model.train()
    return torch.cat(outs_m).numpy(), torch.cat(outs_c).numpy()


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
