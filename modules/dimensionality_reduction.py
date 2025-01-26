import numpy
import scipy.linalg

import utils.plot as P

from .data_visualization import compute_mu_C
from .data_management import vcol, split_classes


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


def execute_PCA(D, m, logger=None):
    mu, C = compute_mu_C(D)
    P = compute_pca(D, m)
    D_PCA = apply_pca(P, D)

    if logger:
        logger.log(f"Compute statistic on dataset:")
        logger.log("mu")
        logger.log(mu)
        logger.log("Covariance matrix = ")
        logger.log(C)
        logger.log()
        logger.log(f"Compute PCA, with m = {m}, P = ")
        logger.log(P)
        logger.log("Apply PCA. D_PCA = ")
        logger.log(D_PCA)
    
    return D_PCA

def visualize_data_PCA(D, L, m, args):
    D0, D1 = split_classes(D, L, 2)

    for dIdx in range(m):
        save_disk = args.save_plots
        output_dir = f"{args.output}/L3_dimensionality_reduction"
        output_name = f"PCA_hist_{dIdx+1}"
        xlabel = f"Direction {dIdx+1}"
        ylabel = 'Relative Frequency'
        P.plot_hist(D0[dIdx, :], D1[dIdx, :], "PCA", xlabel, ylabel, save_disk=save_disk, output_dir=output_dir, output_name=output_name)