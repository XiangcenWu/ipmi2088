import torch



x = torch.tensor([0.0946, 0.5609, 0.3281, 0.2413, 0.4217, 0.1766, 0.3403, 0.0521, 0.0890, 0.3492])



def convert(x):
    x = (x+1.)**3
    return torch.softmax(x, 0)*10


print(convert(x))
