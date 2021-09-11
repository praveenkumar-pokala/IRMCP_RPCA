import numpy as np
import fbpca
import pywt as p
import scipy.io

TOL=1e-7

def converged(Z, d_norm):
    err = np.linalg.norm(Z, 'fro') / d_norm
    return err < TOL

def shrink_mcp(S, tau, gamma):
    t = p.threshold_firm(S, tau, gamma*tau)
    return t

def _svd(M, rank): return fbpca.pca(M, k=min(rank, np.min(M.shape)), raw=True)

def norm_op(M): return _svd(M, 1)[1][0]

def emgn_prox(M, rank, min_sv, gamma):
    u, s, v = _svd(M, rank)
    s = shrink_mcp(s, min_sv, gamma)
    nnz = (s > 0).sum()
    return u @ np.diag(s) @ v, nnz

def core_lsd(X, maxiter, lamda, alpha, gamma, mu, rho): 
    m, n = X.shape
    trans = m<n
    if trans: X = X.T; m, n = X.shape
    
    op_norm = norm_op(X)
   
    Y = np.copy(X) / max(op_norm, np.linalg.norm( X, np.inf) / lamda)

    mu_bar = mu * 1e7
    
    d_norm = np.linalg.norm(X, 'fro')
    L = np.zeros_like(X)
    S = L
    sv = 10
    
    rel_err = np.zeros(1000)
    
    for i in range(int(maxiter)):
        
        X2 = X + Y/mu
       
        S = shrink_mcp(X2 - L, lamda/mu, gamma)

        L, svp = emgn_prox(X2 - S, sv, alpha/mu, gamma)

        sv = svp + (1 if svp < sv else round(0.05*n))

        Z = X - L - S
        Y += mu*Z; mu *= rho
                
        if m > mu_bar: m = mu_bar
        if converged(Z, d_norm): break
    
        d_norm = np.linalg.norm(M1, 'fro')
        err_s = np.linalg.norm(M1-L-S, 'fro') / d_norm
        
        rel_err[i] = err_s
    print('rel_err: ', err_s)
    
    if trans: L=L.T; S=S.T
    return L, S, rel_err

lamda = 0.09783083
alpha = 0.12547759
gamma = 1.277445
mu = 0.00033588
mu_bar = mu * 1e7
rho = 1.12413
maxiter = 1000

m = 300
pr = 0.5
ps = 0.1
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

L1, S1, rel_err =  core_lsd(M1, maxiter, lamda, alpha, gamma, mu, rho)

scipy.io.savemat('rel_err_mcfix.mat', {'rel_err':rel_err})