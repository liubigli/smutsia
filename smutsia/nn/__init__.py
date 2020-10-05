from torch.nn import Linear, Sequential as Seq, BatchNorm1d, LeakyReLU

def MLP(channels, bias=True, negative_slope=0.0):
    return Seq(*[
        Seq(Linear(channels[i - 1], channels[i], bias=bias), BatchNorm1d(channels[i]), LeakyReLU(negative_slope=negative_slope))
        for i in range(1, len(channels))
    ])
