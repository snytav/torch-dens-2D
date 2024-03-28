import torch

def XtoL(x,x0,dh):
	lc = torch.divide(x-x0,dh)
	return lc

def scatter(data,lc,value):
    i  = lc.int()
    di = lc - i
    i = i[0]
    j = i[1]
    di = di[0]
    dj = di[1]

    data[i][j]     += value * (1.0 - di) * (1.0 - dj)
    data[i + 1][j] += value * (di) * (1.0 - dj)
    data[i + 1][j + 1] += value * (di) * (dj)
    data[i][j + 1] += value * (1 - di) * (dj)
    return data

def computeNumberDensity(boxsize,Nx,particles):
    v = 1.0/(particles.shape[0]/Nx[0]/Nx[1]) 
    x0  = torch.zeros(2)
    dh  = torch.divide(boxsize,Nx)
    den = torch.zeros(Nx.int().tolist())
    for part in particles:
        lc = XtoL(part,x0,dh)
        den = scatter(den,lc, v)

    den /= 1.0
    return den

if __name__ == '__main__':
    import numpy as np  # математическая библиотека Python

    boxsize = torch.ones(2)  # размер расчетной области
    Nx = 10*torch.ones(2)  # количество узлов
    N = 100  # количество частиц
    pos = torch.from_numpy(np.random.rand(N, 2)) * boxsize  # массив координат частиц
    #pos = torch.from_numpy(pos)  # преобразоввание в формат библиотеки PyTorch
    pos.requires_grad = True  # флаг, разрешающий взятие производных по массиву pos

    n = computeNumberDensity(boxsize,Nx,pos)
    qq = 0
