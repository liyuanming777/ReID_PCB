import torch.nn as nn
from IPython import embed
class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d,self).__init__()

    def forward(self, x): # x [N,C,H,W]
        inp_size = x.size()
        y = nn.functional.max_pool2d(input = x,kernel_size = (1,inp_size[3]))
        return y


if __name__ == '__main__':
    import  torch
    x = torch.Tensor(32,2048,8,4)
    hp = HorizontalMaxPool2d()
    y = hp(x)
    embed()
