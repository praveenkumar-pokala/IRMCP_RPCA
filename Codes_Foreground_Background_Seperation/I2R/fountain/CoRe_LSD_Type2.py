import numpy as np
import fbpca
from hyperopt import hp, tpe, fmin
from PIL import Image
import cv2
import maxflow
import pywt as p

TOL=1e-7
MAX_ITERS=3
M1 = np.load("../Fountain.npy")
idx = np.load('../idx.npy')
idx = list(idx)
scale = 100
dims = (int(128 * (scale/100)), int(160 * (scale/100)))

def hard(S):
    std = np.std(S)
    O = np.where(np.abs(S) > np.abs(std), 1, 0)
    return O

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
    
    maxiter = 5
    lamda = 0.01741242
    alpha = 4.300637056
    gamma_s = 45.325453
    gamma_l = 85.0669629
    mu = 0.00007243077
    rho = 0.64054019

    op_norm = norm_op(X)

    Y = np.copy(X) / max(op_norm, np.linalg.norm( X, np.inf))
    eta_l = 0

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

        L, svp, wln = emgn_prox(X2 - S, sv, wl, gl)
        gl = gl + eta_l*np.square(wln - wl)
        wl = wln

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

gt_idx = np.array([1157, 1158, 1165, 1179, 1184, 1189, 1190, 1196, 1202, 1204,
                   1422, 1426, 1430, 1440, 1453, 1465, 1477, 1489, 1494, 1509])
iou_score = 0

for i in gt_idx:
    im = postprocessing(np.reshape(M1[:,idx.index(i)], (128, 160)), 
                        np.reshape(S[:,idx.index(i)], (128, 160)))
    gt = Image.open('gt_new_Fountain'+str(int(i))+'.bmp').convert('L')
    gt = np.array(gt)

    intersection = np.logical_and(gt, im)
    union = np.logical_or(gt, im)
    iou_score = iou_score + np.sum(intersection) / np.sum(union)

print('IoU: ', iou_score/20)