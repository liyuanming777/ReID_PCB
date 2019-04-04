import  torch
from IPython import embed
#embed()
def euclidean_dist(x,y):
    '''
    :param x:   pytorch Variable   [m,d]
    :param y:  pytorch Variable   [n,d]
    :return:   pytorch Variable  [m,n]
    '''
    m,n = x.size(0),y.size(0)
    xx = torch.pow(x,2).sum(1,keepdim = True).expand(m,n)
    yy = torch.pow(y,2).sum(1,keepdim = True).expand(n,m).t()
    dist = xx+yy
    dist.addmm_(1,-2,x,y.t())
    dist = dist.clamp(min = 1e-12).sqrt()
    return dist

def batch_euclidean_dist(x,y):
    '''
    :param x:  [batch_size,local_part,Feature channel]
    :param y: [batch_size,local_part,Feature channel]
    :return:    [batch_size,local_part,local_part]

    '''
    assert  len(x.size()) ==3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    N,m,d = x.size()
    N,n,d = y.size()

    #shape [N,m,n]
    xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N,m, n)
    yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, m, n).permute(0,2,1)
    dist = xx + yy
    dist.baddbmm_(1,-2,x,y.permute(0,2,1))
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def shortest_dist(dist_mat):
    m,n = dist_mat.size()[:2]
    dist = [[0 for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            if(i==0) and (j==0):
                dist[i][j] = dist_mat[i][j]
            elif (i==0) and(j>0):
                dist[i][j] = dist[i][j-1]+dist_mat[i][j]
            elif(j==0)and(i>0):
                dist[i][j] = dist[i-1][j]+dist_mat[i][j]
            else:
                dist[i][j] = torch.min(dist[i-1][j] +dist_mat[i][j],dist[i][j-1] +dist_mat[i][j])
    dist  = dist[-1][-1]
    return dist

def MY_shortest_dist(dist_mat):
    m,n = dist_mat.size()[:2]
    dist = [[0 for _ in range(n)] for _ in range(m)]
    k =0
    dist_list=[]
    dist_index = []
    for i in range(m):
        cur = distmat[i][k]
        for j in range(k,n):
            if(cur>distmat[i][j]):
                cur = distmat[i][j]
                k=j
        dist_list.append(cur)
        dist_index.append(k)
    return dist_list,dist_index

if __name__ == '__main__':
    distmat = torch.Tensor([[1,2,3,4],[2,1,3,4],[1,2,3,4],[2,3,4,1]])
    dist ,index= MY_shortest_dist(distmat)
    embed()
