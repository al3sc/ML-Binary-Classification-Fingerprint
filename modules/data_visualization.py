import numpy

import utils.plot as P

from data_management import vcol


def compute_mu_C(D):
    mu = vcol(D.mean(1))
    DC = D - mu
    # C1 = (DC @ (D-mu).T) / float(D.shape[1])
    C = numpy.dot(DC, DC.T) / float(DC.shape[1])
    return mu, C


def visualize_data(D, L, features, save_plots):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    
    # Plot histograms
    for dIdx, feature_name in features.items():
        P.plot_hist(
            D0[dIdx, :], D1[dIdx, :], "Data visualization", feature_name, 
            "Relative frequency", "Fake", "Genuine", 
            save_disk=save_plots, output_dir='./assets/outputs/L2_data_visualization', 
            output_name=f"hist_{dIdx}"
        )

    # Plot pair-wise scatter plots
    for dIdx1, f1 in features.items():
        for dIdx2, f2 in features.items():
            if dIdx1 != dIdx2:
                F0 = [D0[dIdx1, :], D0[dIdx2, :]]
                F1 = [D1[dIdx1, :], D1[dIdx2, :]]
                P.plot_scatter_2_classes(
                    F0, F1, "Data visualization", f1, f2, 
                    "Fake", "Genuine", save_disk=save_plots, 
                    output_dir='./assets/outputs/L2_data_visualization', 
                    output_name=f"scatter_{dIdx1+1}_{dIdx2+1}"
                )

def compute_statistics(D, L, logger=None):
    mu, C = compute_mu_C(D)
    var = D.var(1)
    std = D.std(1)
    
    if logger:
        logger.log_title("Dataset statistics")
        logger.log_paragraph("Means of the features:")
        for i, m in enumerate(mu):
            logger.log(f"Feature {i+1}: {m[0]:.3f}")
        logger.log()

        logger.log_paragraph("Covariance matrix:")
        logger.log_matrix(C)
        logger.log()

        logger.log_paragraph("Variances of the features:")
        for i, v in enumerate(var):
            logger.log(f"Feature {i+1}: {v:.3f}")
        logger.log()

        logger.log_paragraph("Standard deviation of the features:")
        for i, s in enumerate(std):
            logger.log(f"Feature {i+1}: {s:.3f}")
        logger.log()
    
    # Per-class statistics
    for cls in [0, 1]:
        compute_class_statistics(D, L, cls, logger)
    
    return mu, C, var, std

def compute_class_statistics(D, L, cls, logger=None):
    D_Cls = D[:, L == cls]
    mu, C = compute_mu_C(D_Cls)
    var = D_Cls.var(1)
    std = D_Cls.std(1)
    
    if logger:
        logger.log_title(f"Statistics for class {cls}")
        logger.log_paragraph("Means of the features:")
        for i, m in enumerate(mu):
            logger.log(f"Feature {i+1}: {m[0]:.3f}")
        logger.log()

        logger.log_paragraph("Covariance matrix:")
        logger.log_matrix(C)
        logger.log()

        logger.log_paragraph("Variances of the features:")
        for i, v in enumerate(var):
            logger.log(f"Feature {i+1}: {v:.3f}")
        logger.log()

        logger.log_paragraph("Standard deviation of the features:")
        for i, s in enumerate(std):
            logger.log(f"Feature {i+1}: {s:.3f}")
        logger.log()

