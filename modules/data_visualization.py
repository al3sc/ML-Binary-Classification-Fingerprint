import numpy

import utils.plot as P

from .data_management import vcol, split_classes


def compute_mu_C(D):
    mu = vcol(D.mean(1))
    DC = D - mu
    # C1 = (DC @ (D-mu).T) / float(D.shape[1])
    C = numpy.dot(DC, DC.T) / float(DC.shape[1])
    return mu, C


def visualize_data(D, L, features, args):
    D0, D1 = split_classes(D, L, 2)
    
    # Plot histograms
    for dIdx, feature_name in features.items():
        P.plot_hist(
            D0[dIdx, :], D1[dIdx, :], "Data visualization", feature_name, 
            "Relative frequency", "Fake", "Genuine", 
            save_disk=args.save_plots, output_dir=f'{args.output}/L2_data_visualization', 
            output_name=f"hist_{dIdx+1}"
        )

    # Plot pair-wise scatter plots
    for dIdx1, f1 in features.items():
        for dIdx2, f2 in features.items():
            if dIdx1 != dIdx2:
                F0 = [D0[dIdx1, :], D0[dIdx2, :]]
                F1 = [D1[dIdx1, :], D1[dIdx2, :]]
                P.plot_scatter_2_classes(
                    F0, F1, "Data visualization", f1, f2, 
                    "Fake", "Genuine", save_disk=args.save_plots, 
                    output_dir=f'{args.output}/L2_data_visualization', 
                    output_name=f"scatter_{dIdx1+1}_{dIdx2+1}"
                )

def compute_statistics(D, L, logger=None):
    mu, C = compute_mu_C(D)
    var = D.var(1)
    std = D.std(1)
    
    if logger:
        logger.log_paragraph(f"Means of the features:")
        logger.log(f"{mu}\n")
        logger.log_paragraph(f"Covariance matrix:")
        logger.log(f"{C}\n")
        logger.log_paragraph(f"Variances of the features:")
        logger.log(f"{var}\n")    
        logger.log_paragraph(f"Standard deviation of the features:")
        logger.log(f"{std}\n")
    
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
        logger.log_paragraph(f"Means of the features:")
        logger.log(f"{mu}\n")
        logger.log_paragraph(f"Covariance matrix:")
        logger.log(f"{C}\n")
        logger.log_paragraph(f"Variances of the features:")
        logger.log(f"{var}\n")
        logger.log_paragraph(f"Standard deviation of the features:")
        logger.log(f"{std}\n")

