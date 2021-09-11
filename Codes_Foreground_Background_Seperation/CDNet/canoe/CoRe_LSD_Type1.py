import numpy as np
import fbpca
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import maxflow
import pywt as p
import matplotlib.animation as animation

TOL=1e-7
MAX_ITERS=3
dset = 'canoe'
M1 = np.load(dset + ".npy")
gt_arr = np.load(dset + "_gt.npy")
gt_idx = np.load(dset + "_idx.npy")
scale = 100
dims = (int(128 * (scale/100)), int(160 * (scale/100)))

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

def core_lsd_type1(X):
    m, n = X.shape
    trans = m<n
    if trans: X = X.T; m, n = X.shape
    
    op_norm = norm_op(X)
    
    maxiter = 7
    lamda = 0.066*np.ones((m,n))
    alpha = 26.60532*np.ones((m))
    gamma_l = 65.3508*lamda
    gamma_s = 53.68778*alpha
    gamma_lu = 65.3508e7*lamda
    gamma_su = 53.68778e7*alpha
    mu = 2.36765e-6
    rho = 2.701278
    mu_bar = mu * 1e7

    Y = np.copy(X) / max(op_norm, np.linalg.norm( X, np.inf))
        
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

iou_score = 0
count = 0

for i in range(len(gt_arr[0])):
    if i%75 == 0:
        m = np.reshape(M1[:,gt_idx[i]-1], (128, 160))
        s = np.reshape(S[:,gt_idx[i]-1], (128, 160))
        im = postprocessing(m, s)
        gt = np.reshape(gt_arr[:, i], (128, 160))

        intersection = np.logical_and(gt, im)
        union = np.logical_or(gt, im)
        iou_score = iou_score + np.sum(intersection) / np.sum(union)
        count += 1

iou = iou_score/count
print(iou)

# fig = plt.figure(figsize = (8, 10))
# plt.subplot(1, 3, 1)
# plt.imshow(m, cmap = 'gray')
# plt.axis('Off')
# plt.subplot(1, 3, 2)
# plt.imshow(im, cmap = 'gray')
# plt.axis('Off')
# plt.subplot(1, 3, 3)
# plt.imshow(gt, cmap = 'gray')
# plt.axis('Off')
# plt.show()
# np.save('core_lsd_type1_L.npy', np.reshape(np.transpose(L), (-1, 128, 160)))
# np.save('core_lsd_type1_S.npy', np.reshape(np.transpose(S), (-1, 128, 160)))

# fig = plt.figure()
# ims = []
# for i in range(np.shape(M1)[1]):
#     act = im = np.reshape(M1[:,i], (128, 160))
#     pp_img = postprocessing(np.reshape(M1[:,i], (128, 160)), 
#                             np.reshape(S[:,i], (128, 160)))
#     im = plt.imshow(np.multiply(act, pp_img), cmap = 'gray', animated = True)
#     ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000)
# plt.axis('Off')
# ani.save('core_lsd_type1.mp4')