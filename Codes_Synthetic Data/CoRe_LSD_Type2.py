import numpy as np
import fbpca
import pywt as p
import scipy.io

TOL=1e-7
#np.random.seed(0)
def converged(Z, d_norm):
    err = np.linalg.norm(Z, 'fro') / d_norm
    return err < TOL

def emcp_prox(M, ws, mu, gs):
    S = np.where(np.abs(M)-ws/mu > 0, np.abs(M)-ws/mu, 0)*np.sign(M)
    
    ws = np.where(ws - np.divide(S, gs) > 0, ws - np.divide(S, gs), 0) 
    return S, ws

def _svd(M, rank): return fbpca.pca(M, k=min(rank, np.min(M.shape)), raw=True)
def norm_op(M): return _svd(M, 1)[1][0]
#def emcp_prox(S, tau, gamma): return p.threshold_firm(S, tau, gamma*tau)

def emgn_prox(M, rank, w, gl, mu):
    
    u, s, v = _svd(M, rank)
    
    s -= w[:len(s)]/mu
    nnz = (s > 0).sum()
    s = np.where(s>0, s, 0)
    w[:nnz] = np.where(w[:nnz] - np.divide(s[:nnz],gl[:nnz]) > 0, w[:nnz] - np.divide(s[:nnz],gl[:nnz]), 0) 
    
    return u[:,:nnz] @ np.diag(s[:nnz]) @ v[:nnz], nnz, w

def core_lsd_type2(X, maxiter, lamda, alpha, gamma_s, gamma_l, mu, rho):
    m, n = X.shape
    trans = m<n
    if trans: X = X.T; m, n = X.shape
    
    op_norm = norm_op(X)
    mu_bar = mu * 1e7
    
    Y = np.copy(X) / max(op_norm, np.linalg.norm( X, np.inf))
    
    d_norm = np.linalg.norm(X, 'fro')
    L = np.zeros_like(X)
    S = L
    sv = 10

    ws = np.ones_like(X)*(lamda)
    gs = np.ones_like(X)*(gamma_s)
    wl = np.ones(n)*(alpha)
    gl = np.ones(n)*gamma_l
    
    rel_err = np.zeros(1000)
    
    for i in range(int(maxiter)):
        
        X2 = X + Y/mu

        S, ws = emcp_prox(X2 - L, ws, mu, gs)        

        L, svp, wl = emgn_prox(X2 - S, sv, wl, gl, mu)

        sv = svp + (1 if svp < sv else round(0.05*n))
        sv = min(sv, n)

        Z = X - L - S
        Y += mu*Z; mu *= rho
        mu = min(mu, mu_bar)
        
        
        if m > mu_bar: m = mu_bar
        #if converged(Z, d_norm): break
    
        d_norm = np.linalg.norm(M1, 'fro')
        err_s = np.linalg.norm(M1-L-S, 'fro') / d_norm
        
        rel_err[i] = err_s
        
    print('rel_err: ', err_s)
    
    if trans: L=L.T; S=S.T
    return L, S, rel_err

lamda = 0.3418177874
alpha = 0.050583116
gamma_s = 24.5239812951
gamma_l = 97.98199267
mu = 0.405669521
rho = 5.18001883
maxiter = 1000

m = 300
pr = 0.5
ps = 0.5
r = int(np.round(pr*m))
print('Rank: ', r)
sn = int(np.round(ps*m*m))

U = np.random.randn(m, r)
V = np.random.randn(m, r)
L = U @ V.T

S = np.zeros_like(L)
ind = np.random.randint(0, m*m, sn)
S = np.reshape(S, m*m)
S[ind] = 1
S = np.reshape(S, (m, m))
S = S*(100*(np.random.rand(m,m)-50))

M1 = L + S

L1, S1, rel_err =  core_lsd_type2(M1, maxiter, lamda, alpha, gamma_s, gamma_l, mu, rho)

scipy.io.savemat('rel_err_mcvar_new.mat', {'rel_err':rel_err})