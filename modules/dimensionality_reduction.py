import numpy
import scipy.linalg

import utils.plot as P

from .data_visualization import compute_mu_C
from .data_management import vcol, vrow, split_classes, save_csv, compute_error_rates_multi


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
    P = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = numpy.dot(P, numpy.dot(Sb, P.T))

    U2, s2, _ = numpy.linalg.svd(Sb2)

    P2 = U2[:, 0:m]     # m highest eigenvalues of the matrix corresponding to the eigenvectors of Sb2
    return numpy.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D


# Apply PCA and visualize directions histograms

def execute_PCA(D, m, logger=None, DVAL=None):
    P = compute_pca(D, m)
    D_PCA = apply_pca(P, D)
    if DVAL is not None:
        DVAL_PCA = apply_pca(P, DVAL)

    if logger:
        logger.log_paragraph(f"Compute PCA, with m = {m}")
        logger.log(f"P:\n{P}\n")
        logger.log(f"Apply PCA. D_PCA:\n{D_PCA}\n")
    
    return D_PCA, DVAL_PCA if DVAL is not None else None

def visualize_data_PCA(D, L, m, args):
    D0, D1 = split_classes(D, L, 2)

    # Plot histograms
    for dIdx in range(m):
        save_disk = args.save_plots
        output_dir = f"{args.output}/L3_dimensionality_reduction"
        output_name = f"PCA_hist_{dIdx+1}"
        xlabel = f"Direction {dIdx+1}"
        ylabel = 'Relative Frequency'
        P.plot_hist(D0[dIdx, :], D1[dIdx, :], "PCA", xlabel, ylabel, save_disk=save_disk, output_dir=output_dir, output_name=output_name)
    
    # Plot pair-wise scatter plots
    for dIdx1 in range(m):
        for dIdx2 in range(m):
            if dIdx1 != dIdx2:
                F0 = [D0[dIdx1, :], D0[dIdx2, :]]
                F1 = [D1[dIdx1, :], D1[dIdx2, :]]
                P.plot_scatter_2_classes(
                    F0, F1, "Data visualization", f"Direction {dIdx1+1}", f"Direction {dIdx2+1}", 
                    "Fake", "Genuine", save_disk=args.save_plots, 
                    output_dir=f'{args.output}/L3_dimensionality_reduction', 
                    output_name=f"PCA_scatter_{dIdx1+1}_{dIdx2+1}"
                )

# Apply LDA and visualize histograms

def execute_LDA(D, L, logger=None):
    U = compute_lda_geig(D, L, m = 1)
    V = compute_lda_JointDiag(D, L, m = 1)

    DU = apply_lda(-U, D)

    if logger:
        logger.log_paragraph("Computing LDA using the eigenvalues approach.")
        logger.log(f"U:\n{U}")
        logger.log_paragraph("Computing LDA using the joint diagonalization approach.")
        logger.log(f"V:\n{V}")
        logger.log_paragraph("Apply LDA:")
        logger.log(f"{DU}\n")

    return DU

def visualize_data_LDA(D, L, args, title="LDA"):
    D0, D1 = split_classes(D, L, 2)

    # Plot histogram
    save_disk = args.save_plots
    output_dir = f"{args.output}/L3_dimensionality_reduction"
    output_name = f"{title.replace(" ", "_")}_hist"
    xlabel = "LDA Direction"
    ylabel = 'Relative Frequency'
    P.plot_hist(D0[0], D1[0], title, xlabel, ylabel, save_disk=save_disk, output_dir=output_dir, output_name=output_name)


# Apply LDA on training and validation set
def execute_LDA_TrVal(DTR, LTR, DVAL, logger=None):
    U = compute_lda_geig(DTR, LTR, m=1)     # train the model (on the training data)
    
    DTR_LDA = apply_lda(U, DTR)

    if logger:
        logger.log_paragraph("Computing LDA using the eigenvalues approach.")
        logger.log(f"U:\n{U}\n")
        logger.log(f"Apply LDA:\n{DTR_LDA}\n")

    # Check if the Fake class samples are, on average, on the right of the Genuine samples on the training set. If not, we reverse ULDA and re-apply the transformation.
    if DTR_LDA[0, LTR==0].mean() > DTR_LDA[0, LTR==1].mean():
        U = -U
        DTR_LDA = apply_lda(U, DTR)

        if logger:
            logger.log("Reverse U and re-apply LDA on training data.\nThe Fake class samples are not, on average, on the right of the Genuine samples.")
            logger.log(f"New Fake class means:\n{DTR_LDA[0, LTR==0].mean()}\n")
            logger.log(f"New Genuine class means:\n{DTR_LDA[0, LTR==1].mean()}\n")
            
    DVAL_LDA = apply_lda(U, DVAL)

    if logger:
        logger.log(f"Apply LDA on validation data:\n{DVAL_LDA}\n")

    return DTR_LDA, DVAL_LDA


# Apply LDA as a classifier
def classify_LDA(DTR, LTR, DVAL, LVAL, logger=None, offset=None):
    threshold = (DTR[0,LTR==0].mean() + DTR[0,LTR==1].mean()) / 2.0 # Projected samples have only 1 dimension
    if offset:
        threshold += offset
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL[0] >= threshold] = 1
    PVAL[DVAL[0] < threshold] = 0
    
    if logger:
        logger.log(f"Computed threshold (1-D): {threshold}\n")
        logger.log(f'Labels:      {LVAL}')
        logger.log(f'Predictions: {PVAL}')
        logger.log(f'Number of erros: {(PVAL != LVAL).sum()} (out of {LVAL.size} samples)')
        logger.log(f'Error rate: {( (PVAL != LVAL).sum() / float(LVAL.size) *100 ):.2f}%')
        logger.log(f'Accuracy: {( (PVAL == LVAL).sum() / float(LVAL.size) *100 ):.2f}%')

    return PVAL

def classify_LDA_manyThresholds(DTR, LTR, DVAL, LVAL, logger=None, save_tables=None):
    th_offsets = [-0.2, -0.1, -0.05, +0.05, +0.1, +0.2]
    PVALs = []
    
    DTR_LDA, DVAL_LDA = execute_LDA_TrVal(DTR, LTR, DVAL)
    for o in th_offsets:
        if logger:
            logger.log_paragraph(f"Threshold offset: {o}")
        PVAL = classify_LDA(DTR_LDA, LTR, DVAL_LDA, LVAL, logger, o)
        PVALs.append(PVAL)

    if save_tables:
        header = ["Thresholds", "-0.2", "-0.1", "-0.05", "+0.05", "+0.1", "+0.2"]
        row = ["Error rates", *[format(x, ".2f") for x in compute_error_rates_multi(PVALs, LVAL)]]
        save_csv([row], header, logger, "LDA_manyThresholds", "L3_dimensionality_reduction")


    return PVALs


# Pre-process features with PCA before applying LDA as a classifier
def classify_LDA_prePCA(DTR, LTR, DVAL, LVAL, directions, logger=None, save_tables=None):
    PVALs = []

    for m in range(directions):
        if logger:
            logger.log_paragraph(f"Pre processing features with {m+1} PCA directions.")
        
        DTR_PCA, DVAL_PCA = execute_PCA(DTR, m+1, DVAL=DVAL)
        DTR_LDA, DVAL_LDA = execute_LDA_TrVal(DTR_PCA, LTR, DVAL_PCA)

        PVAL = classify_LDA(DTR_LDA, LTR, DVAL_LDA, LVAL, logger)
        PVALs.append(PVAL)
    
    if save_tables:
        header = ["PCA Directions", *[f"{m+1}" for m in range(directions)]]
        row = ["Error rates", *[format(x, ".2f") for x in compute_error_rates_multi(PVALs, LVAL)]]
        save_csv([row], header, logger, "LDA_prePCA", "L3_dimensionality_reduction")

    
    return PVALs