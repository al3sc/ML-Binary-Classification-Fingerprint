import numpy

from .data_management import vrow
from .data_visualization import compute_mu_C

import utils.plot as P


def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)     # precision matrix
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

def compute_ll(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

def fit_univariate_Gaussian_toFeatures(D, L, save_disk, logger):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    m0_ML, C0_ML = compute_mu_C(D0)
    m1_ML, C1_ML = compute_mu_C(D1)

    if logger:
        logger.log_paragraph("Computing means and covariance matrices for each class")
        logger.log(f"Class 0 mean: {m0_ML}")
        logger.log(f"Class 1 mean: {m1_ML}")
        logger.log(f"Class 0 covariance matrix: {C0_ML}")
        logger.log(f"Class 1 covariance matrix: {C1_ML}")     


    for f in range(6):
        feat0 = D0[f, :]
        feat1 = D1[f, :]

        # Gaussian pdfs

        # Class 0 - Fake
        x0_vals = numpy.linspace(min(feat0), max(feat0), 100)
        m0_feat = m0_ML[f]
        C0_feat = numpy.array([[C0_ML[f, f]]])      # 1x1 variance array
        pdf0 = numpy.exp(logpdf_GAU_ND(vrow(x0_vals), m0_feat, C0_feat))
        P.plot_Gaussian_density(
            feat0, x0_vals, pdf0, "Distribution density over histogram", f"Feature {f+1}",
            "Relative Frequency", "Fake", "blue", "orange", nBins=50, save_disk = save_disk,
            output_dir="./assets/outputs/L4_gaussian_density_estimation", output_name=f"hist_gaussian_Fake_{f+1}"
        )


        # Class 1 - Genuine
        x1_vals = numpy.linspace(min(feat1), max(feat1), 100)
        m1_feat = m1_ML[f]
        C1_feat = numpy.array([[C1_ML[f, f]]])      # 1x1 variance array
        pdf1 = numpy.exp(logpdf_GAU_ND(vrow(x1_vals), m1_feat, C1_feat))
        P.plot_Gaussian_density(
            feat1, x1_vals, pdf1, "Distribution density over histogram", f"Feature {f+1}", 
            "Relative Frequency", "Genuine", "orange", "blue", nBins=50, save_disk = save_disk,
            output_dir="./assets/outputs/L4_gaussian_density_estimation", output_name=f"hist_gaussian_Genuine_{f+1}"
        )

        if logger:
            logger.log_paragraph(f"Feature {f+1}")
            logger.log("### CLASS 0")
            logger.log(f"Values for X: {x0_vals}")
            logger.log(f"Feature's mean: {m0_feat}")
            logger.log(f"Feature's covariance matrix: {C0_feat}")
            logger.log(f"Probability Density Function values: {pdf0}")
            logger.log("\n### CLASS 1")
            logger.log(f"Values for X: {x1_vals}")
            logger.log(f"Feature's mean: {m1_feat}")
            logger.log(f"Feature's covariance matrix: {C1_feat}")
            logger.log(f"Probability Density Function values: {pdf1}")
            