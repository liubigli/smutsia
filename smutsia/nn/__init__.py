from torch.nn import Linear, Sequential as Seq, BatchNorm1d, ReLU

def MLP(channels):
    return Seq(*[
        Seq(Linear(channels[i - 1], channels[i]), BatchNorm1d(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])
