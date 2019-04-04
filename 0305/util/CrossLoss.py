import  torch
from torch import nn
from IPython import embed
def cross_entropy(input_, target, reduction='elementwise_mean'):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    res  =-target * logsoftmax(input_)
    if reduction == 'elementwise_mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return res

def pidstoOne_hot(pids,class_num):
    targets = torch.zeros(pids.size()[0], class_num).scatter_(1, pids.data.cpu(), 1)
    return targets

    #targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

if __name__ =='__main__':
    input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
    label = torch.LongTensor(2, 1).random_()% 3
    #target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    target = pidstoOne_hot(label,3)
    N_norm = torch.FloatTensor([0.4,0.6])
    target = target*(N_norm.reshape([2,1]).expand_as(target))
    embed()
    loss = cross_entropy(input, target)