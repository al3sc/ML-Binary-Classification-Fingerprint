import numpy
import scipy.linalg

from data_visualization import compute_mu_C
from data_management import vcol


### PCA - Principal Component Analysis ###

def compute_pca(D, m):

    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D


### LDA - Linear Discriminant Analysis ###

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

# generalized eigenvalue problem: Sb*w = λ*Sw*w
def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)    # solve the  generalized problem for hermitian matrices
    W = U[:, ::-1][:, 0:m]
    return W

# joint diagonalization: P1 = UΣ^(−1/2)U.T
def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = numpy.linalg.svd(Sw)
    P = numpy.dot(U * pm.vrow(1.0/(s**0.5)), U.T)

    Sb2 = numpy.dot(P, numpy.dot(Sb, P.T))

    U2, s2, _ = numpy.linalg.svd(Sb2)

    P2 = U2[:, 0:m]     # m highest eigenvalues of the matrix corresponding to the eigenvectors of Sb2
    return numpy.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D

