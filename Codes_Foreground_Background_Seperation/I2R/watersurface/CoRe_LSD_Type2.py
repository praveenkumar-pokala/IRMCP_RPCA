import numpy as np
import fbpca
from hyperopt import hp, tpe, fmin
from PIL import Image
import cv2
import maxflow
import pywt as p
from matplotlib import pyplot as plt

TOL=1e-7
MAX_ITERS=3
M1 = np.load("WaterSurface.npy")
scale = 100
dims = (int(128 * (scale/100)), int(160 * (scale/100)))

def hard(S):
    std = np.std(S)
    O = np.where(np.abs(S) > np.abs(std), 1, 0)
    return O

def postprocessing1(im, unary):
        unary = np.float32(unary)
        unary = cv2.GaussianBlur(unary, (3,3), 0)

        g = maxflow.Graph[float]()
        nodes = g.add_grid_nodes(unary.shape)

        for i in np.arange(im.shape[0]):
                for j in np.arange(im.shape[1]):
                        v = nodes[i,j]
                        g.add_tedge(v, -unary[i,j], -1.0+unary[i,j])

        def potts_add_edge(i0, j0, i1, j1):
                v0, v1 = nodes[i0,j0], nodes[i1,j1]
                w = 0.1 * np.exp(-((im[i0,j0] - im[i1,j1])**2).sum() / 0.1)
                g.add_edge(v0, v1, w, w)

        for i in np.arange(1,im.shape[0]-1):
                for j in np.arange(1,im.shape[1]-1):
                        potts_add_edge(i, j, i, j-1)
                        potts_add_edge(i, j, i, j+1)
                        potts_add_edge(i, j, i-1, j)
                        potts_add_edge(i, j, i+1, j)

        g.maxflow()
        seg = np.float32(g.get_grid_segments(nodes))
        return seg
    
def postprocessing(im, unary):
        unary = np.float32(unary)
        unary = cv2.GaussianBlur(unary, (9,9), 0)

        g = maxflow.Graph[float]()
        nodes = g.add_grid_nodes(unary.shape)

        for i in np.arange(im.shape[0]):
                for j in np.arange(im.shape[1]):
                        v = nodes[i,j]
                        g.add_tedge(v, -unary[i,j], -1.0+unary[i,j])

        def potts_add_edge(i0, j0, i1, j1):
                v0, v1 = nodes[i0,j0], nodes[i1,j1]
                w = 0.1 * np.exp(-((im[i0,j0] - im[i1,j1])**2).sum() / 0.1)
                g.add_edge(v0, v1, w, w)

        for i in np.arange(1,im.shape[0]-1):
                for j in np.arange(1,im.shape[1]-1):
                        potts_add_edge(i, j, i, j-1)
                        potts_add_edge(i, j, i, j+1)
                        potts_add_edge(i, j, i-1, j)
                        potts_add_edge(i, j, i+1, j)

        g.maxflow()
        seg = np.float32(g.get_grid_segments(nodes))
        return seg

def emcp_prox(S, tau, gamma):
    t = p.threshold_firm(S, tau, gamma*tau)
    return t

def converged(Z, d_norm):
    err = np.linalg.norm(Z, 'fro') / d_norm

    return err < TOL

def shrink(M, w):
    S = np.where(np.abs(M)-w > 0, np.abs(M)-w, 0)*np.sign(M)
    return S


def _svd(M, rank): return fbpca.pca(M, k=min(rank, np.min(M.shape)), raw=True)


def norm_op(M): return _svd(M, 1)[1][0]


def emgn_prox(M, rank, w, gl):

    u, s, v = _svd(M, rank)

    s -= w[:len(s)]
    nnz = (s > 0).sum()
    s = np.where(s>0, s, 0)
    w[:nnz] = np.where(w[:nnz] - np.divide(s[:nnz],gl[:nnz]) > 0, w[:nnz] - np.divide(s[:nnz],gl[:nnz]), 0)

    return u[:,:nnz] @ np.diag(s[:nnz]) @ v[:nnz], nnz, w

def core_lsd_type2(X): # refactored
    m, n = X.shape
    trans = m<n
    if trans: X = X.T; m, n = X.shape
    
    maxiter = 8
    lamda = 0.025173722
    alpha = 65.2377469
    gamma_s = 76.154997689
    gamma_l = 8.9109064
    mu = 0.000240304689
    rho = 0.85816017
#    rho = 1

    op_norm = norm_op(X)

    Y = np.copy(X) / max(op_norm, np.linalg.norm( X, np.inf))

    d_norm = np.linalg.norm(X, 'fro')
    L = np.zeros_like(X)
    sv = 10
    mu_bar = mu*1e7

    ws = lamda/mu
    wl = np.ones(n)*(alpha/mu)
    gs = gamma_s
    gl = np.ones(n)*gamma_l    
    
    for i in range(int(maxiter)):

        X2 = X + Y/mu

        S = emcp_prox(X2 - L, ws, ws*gs)

        L, svp, wl = emgn_prox(X2 - S, sv, wl, gl)

        sv = svp + (1 if svp < sv else round(0.05*n))
        sv = min(sv, n)

        Z = X - L - S
        Y += mu*Z; mu *= rho
        mu = min(mu, mu_bar)


        if m > mu_bar: m = mu_bar
        if converged(Z, d_norm): break

    if trans: L=L.T; S=S.T
    return L, S


L, S =  core_lsd_type2(M1)

B = S

S = np.where(np.abs(S)==0, 0, 1)

gt_idx = np.array([499, 515, 523, 547, 548, 553, 554, 559, 575, 577, 
                   594, 597, 601, 605, 615, 616, 619, 620, 621, 624])
iou_score = 0

for i in gt_idx:
    im = postprocessing1(np.reshape(M1[:,i], (128, 160)), postprocessing(np.reshape(M1[:,i], (128, 160)), 
                        np.reshape(S[:,i], (128, 160)))*np.reshape(S[:,i], (128, 160)))
#    im = np.reshape(S[:,i], (128, 160))
    gt = Image.open('gt_new_WaterSurface1'+str(int(i))+'.bmp').convert('L')
    gt = np.array(gt)

    intersection = np.logical_and(gt, im)
    union = np.logical_or(gt, im)
    iou_score = iou_score + np.sum(intersection) / np.sum(union)


# fig = plt.figure(figsize = (8, 10))
# plt.subplot(1, 3, 1)
# plt.imshow(np.reshape(M1[:, 624], (128, 160)), cmap = 'gray')
# plt.axis('Off')
# plt.subplot(1, 3, 2)
# plt.imshow(np.reshape(L[:, 624], (128, 160)), cmap = 'gray')
# plt.axis('Off')
# plt.subplot(1, 3, 3)
# plt.imshow(im, cmap = 'gray')
# plt.axis('Off')
# plt.show()

print('IoU: ', iou_score/20)