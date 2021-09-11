import numpy as np
import fbpca
import matplotlib.pyplot as plt
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

def postprocessing(im, unary):
	unary = np.float32(unary)
	unary = cv2.GaussianBlur(unary, (7,7), 0)

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
       
def converged(Z, d_norm):
    err = np.linalg.norm(Z, 'fro') / d_norm
#    print('error: ', err)
    return err < TOL

def firm_threshold(x, value_low, value_high):
    x = np.where(np.abs(x) <= value_low, 0, x)
    x = np.where(np.logical_and(value_low < np.abs(x), np.abs(x) <= value_high), 
                 x * value_high * (1 - value_low/np.abs(x))/(value_high - value_low), x)
    
    return x

def emcp_prox(S, tau, gamma):
    t = firm_threshold(S, tau, gamma)
    return t

def _svd(M, rank): return fbpca.pca(M, k=min(rank, np.min(M.shape)), raw=True)


def norm_op(M): return _svd(M, 1)[1][0]

def emgn_prox(M, rank, alpha, gamma, gamma_u, mu):
    min_sv = alpha/mu
    gamma_sv = gamma/mu
    u, s, v = _svd(M, rank)
    nnz = (s > 0).sum()
    s = emcp_prox(s[:nnz], min_sv[:nnz], gamma_sv[:nnz])
    alpha[:nnz] = np.where(alpha[:nnz] - np.divide(np.abs(s), gamma_u[:nnz])>0
                     , alpha[:nnz] - np.divide(np.abs(s), gamma_u[:nnz]), 0)
    nnz = (s > 0).sum()
    return u[:, :nnz] @ np.diag(s[:nnz]) @ v[:nnz], nnz, alpha

def core_lsd_type1(X): # refactored
    m, n = X.shape
    trans = m<n
    if trans: X = X.T; m, n = X.shape
    
    op_norm = norm_op(X)
    
    maxiter = 10
    
    lamda = 0.024769*np.ones((m,n))
    alpha = 6.9912*np.ones((m))
    gamma_s = 63.9473*lamda
    gamma_l = 63.9473*alpha
    gamma_su = 63.9473e7*lamda
    gamma_lu = 63.9473e7*alpha
   
    Y = np.copy(X) / max(op_norm, np.linalg.norm( X, np.inf))
    
    mu = 0.0007524
    mu_bar = mu * 1e7
    rho = 0.75695
    
    d_norm = np.linalg.norm(X, 'fro')
    L = np.zeros_like(X)
    sv = 10    
    
    for i in range(maxiter):
        
        X2 = X + Y/mu

        S = emcp_prox(X2 - L, lamda/mu, gamma_s/mu)
        lamda = np.where(lamda - np.divide(np.abs(S), gamma_su)>0
                         , lamda - np.divide(np.abs(S), gamma_su), 0)

        L, svp, alpha = emgn_prox(X2 - S, sv, alpha, gamma_l, gamma_lu, mu)

        sv = svp + (1 if svp < sv else round(0.05*n))

        Z = X - L - S
        Y += mu*Z; mu *= rho

        if m > mu_bar: m = mu_bar
        if converged(Z, d_norm): break
    
    if trans: L=L.T; S=S.T
    return L, S


L, S =  core_lsd_type1(M1)
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

# fig = plt.figure(figsize = (8, 10))
# plt.subplot(1, 3, 1)
# plt.imshow(np.reshape(M1[:, idx.index(1509)], (128, 160)), cmap = 'gray')
# plt.axis('Off')
# plt.subplot(1, 3, 2)
# plt.imshow(np.reshape(L[:, idx.index(1509)], (128, 160)), cmap = 'gray')
# plt.axis('Off')
# plt.subplot(1, 3, 3)
# plt.imshow(im, cmap = 'gray')
# plt.axis('Off')
# plt.show()
print('IoU: ', iou_score/20)