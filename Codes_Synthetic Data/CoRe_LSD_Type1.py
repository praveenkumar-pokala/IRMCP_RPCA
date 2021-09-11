import numpy as np
import fbpca
import scipy.io

TOL=1e-7

def converged(Z, d_norm):
    err = np.linalg.norm(Z, 'fro') / d_norm
    return err < TOL

def firm_threshold(x, value_low, value_high):
    x = np.where(np.abs(x) <= value_low, 0, x)
    x = np.where(np.logical_and(value_low < np.abs(x), np.abs(x) <= value_high), 
                 x * value_high * (1 - value_low/x)/(value_high - value_low), x)
    
    return x

def emcp_prox(S, tau, gamma):
    t = firm_threshold(S, tau, gamma)
    return t

def _svd(M, rank): return fbpca.pca(M, k=min(rank, np.min(M.shape)), raw=True)

def norm_op(M): return _svd(M, 1)[1][0]

def emgn_prox(M, rank, alpha, gamma, mu):
    min_sv = alpha/mu
    gamma_sv = gamma/mu
    u, s, v = _svd(M, rank)
    nnz = (s > 0).sum()
    s = emcp_prox(s[:nnz], min_sv[:nnz], gamma_sv[:nnz])
    alpha[:nnz] = np.where(alpha[:nnz] - np.divide(np.abs(s), gamma[:nnz])>0
                     , alpha[:nnz] - np.divide(np.abs(s), gamma[:nnz]), 0)
    nnz = (s > 0).sum()
    return u[:, :nnz] @ np.diag(s[:nnz]) @ v[:nnz], nnz, alpha

def core_lsd_type1(X, maxiter, lamda, alpha, gamma_l, gamma_s, mu, rho): 
    m, n = X.shape
    trans = m<n
    if trans: X = X.T; m, n = X.shape
    
    op_norm = norm_op(X)
   
    Y = np.copy(X) / max(op_norm, np.linalg.norm( X, np.inf) / np.mean(lamda))

    mu_bar = mu * 1e7
    
    d_norm = np.linalg.norm(X, 'fro')
    L = np.zeros_like(X)
    S = L
    sv = 10
    
    rel_err = np.zeros(1000)
    
    for i in range(int(maxiter)):
        
        X2 = X + Y/mu
       
        S = emcp_prox(X2 - L, lamda/mu, gamma_s/mu)
        lamda = np.where(lamda - np.divide(np.abs(S), gamma_s)>0
                         , lamda - np.divide(np.abs(S), gamma_s), 0)

        L, svp, alpha = emgn_prox(X2 - S, sv, alpha, gamma_l, mu)

        sv = svp + (1 if svp < sv else round(0.05*n))

        Z = X - L - S
        Y += mu*Z; mu *= rho
                
        if m > mu_bar: m = mu_bar
        #if converged(Z, d_norm): break
    
        d_norm = np.linalg.norm(M1, 'fro')
        err_s = np.linalg.norm(M1-L-S, 'fro') / d_norm
        
        rel_err[i] = err_s
    print('rel_err: ', err_s)
    
    if trans: L=L.T; S=S.T
    return L, S, rel_err

m = 300
pr = 0.5
ps = 0.5

lamda = 0.598919202*np.ones((m,m))
alpha = 3.578871927*np.ones((m))
gamma_s = 13.89197073*np.ones((m,m))
gamma_l = 1.017366686*np.ones((m))
mu = 0.0043398162
rho = 1.197774963
maxiter = 1000

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

L1, S1, rel_err =  core_lsd_type1(M1, maxiter, lamda, alpha, gamma_l, gamma_s, mu, rho)

scipy.io.savemat('rel_err_mcvar_lamda_1.mat', {'rel_err':rel_err})