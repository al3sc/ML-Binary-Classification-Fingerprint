import numpy
import scipy.special

from .data_visualization import compute_mu_C
from .data_management import vcol, vrow, compute_error_rates_multi, save_csv
from .dimensionality_reduction import execute_PCA
from .gaussian_density_estimation import logpdf_GAU_ND

# Compute a dictionary of ML parameters for each class
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

# Compute a dictionary of ML parameters for each class - Naive Bayes version of the model
# We compute the full covariance matrix and then extract the diagonal. Efficient implementations would work directly with just the vector of variances (diagonal of the covariance matrix)
def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * numpy.eye(D.shape[0]))          # multiply by the identity matrix -> diagonal covariance matrix 
    return hParams

# Compute a dictionary of ML parameters for each class - Tied Gaussian model
# We exploit the fact that the within-class covariance matrix is a weighted mean of the covraince matrices of the different classes
def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C_class = compute_mu_C(DX)
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    CGlobal = CGlobal / D.shape[1]
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams


# Compute per-class log-densities. We assume classes are labeled from 0 to C-1. The parameters of each class are in hParams (for class i, hParams[i] -> (mean, cov))
def compute_log_likelihood_Gau(D, hParams):

    S = numpy.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])          # for each sample of the class lab, the value associated to the log-density
    return S

# compute log-postorior matrix from log-likelihood matrix and prior array
def compute_logPosterior(S_logLikelihood, v_prior):
    logSJoint = S_logLikelihood + vcol(numpy.log(v_prior))                      # joint log-density 
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))             # marginal log-density
    logSPost = logSJoint - logSMarginal                                         # log-posteriors
    return logSPost



def binaryMVGModels(name, DTR, LTR, DVAL, LVAL, ML_func=Gau_MVG_ML_estimates, prior=1/2, logger=None):
    hParams = ML_func(DTR, LTR)

    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams)

    llr = S_logLikelihood[1] - S_logLikelihood[0]
    threshold = -numpy.log( prior / (1 - prior) )             # 0 if prior == 1/2

    predictions_class1 = numpy.array(llr >= threshold)

    n_correct_predictions = numpy.sum(predictions_class1 == LVAL)
    n_wrong_predictions = numpy.sum(predictions_class1 != LVAL)

    accuracy = n_correct_predictions/DVAL.shape[1]
    error_rate = n_wrong_predictions/DVAL.shape[1]


    if logger:
        logger.log_paragraph(f'Model: {name}')
        logger.log(f'Total number of samples: {DVAL.shape[1]}')
        logger.log(f'Number of correct predictions: {n_correct_predictions}')
        logger.log(f'Number of wrong predictions: {n_wrong_predictions}')
        logger.log(f'Accuracy: {accuracy*100:.2f}%')
        logger.log(f'Error rate: {error_rate*100:.2f}%\n')
    
    return predictions_class1


def apply_all_binary_MVG(DTR, LTR, DVAL, LVAL, logger=None, save_tables=None):

    # MVG
    PVAL_MVG = binaryMVGModels("MVG", DTR, LTR, DVAL, LVAL, ML_func=Gau_MVG_ML_estimates, logger=logger)

    # MVG Tied
    PVAL_Tied = binaryMVGModels("Tied MVG", DTR, LTR, DVAL, LVAL, ML_func=Gau_Tied_ML_estimates, logger=logger)

    # MVG Naive Bayes
    PVAL_Naive = binaryMVGModels("Naive Bayes MVG", DTR, LTR, DVAL, LVAL, ML_func=Gau_Naive_ML_estimates, logger=logger)

    if save_tables:
        header = ["Models", "MVG", "Tied MVG", "Naive Bayes MVG"]
        PVALs = [PVAL_MVG, PVAL_Tied, PVAL_Naive]
        row = ["Error rates", *[format(x, ".2f") for x in compute_error_rates_multi(PVALs, LVAL)]]

        save_csv([row], header, logger, "All_MVG", "L5_generative_models")

def analyze_C_MVG_models(DTR, LTR, logger=None):
    funcs = [Gau_MVG_ML_estimates, Gau_Tied_ML_estimates, Gau_Naive_ML_estimates]
    models = ["MVG", "Tied MVG", "Naive Bayes MVG"]
    classes = ['Fake', 'Genuine']

    for m, f in zip(models, funcs):
        logger and logger.log_paragraph(f"{m} results analysis")
        hParams = f(DTR, LTR)

        Cs = [ hParams[0][1], hParams[1][1] ]
        
        for i, label in enumerate(classes):
            logger and logger.info(f'Class {label}')
            C = Cs[i]
            for dIdx1 in range(C.shape[0]):
                feat_covariances = []
                variance = C[dIdx1][dIdx1]

                for dIdx2 in range(C.shape[1]):
                    if(dIdx1 != dIdx2):
                        covariance = C[dIdx1][dIdx2]
                        feat_covariances.append( (dIdx2+1, covariance, (variance - covariance)) )
                    
                logger and logger.log(f'\nFeature {dIdx1+1} variance: {variance}.\n')
                for cov_info in feat_covariances:
                    logger and logger.log(f'\tCovariance with feature {cov_info[0]}: {cov_info[1]} \t(Variance - Covariance: {cov_info[2]:.3f})')
            logger and logger.log()

def compute_Pearson_correlation(DTR, LTR, logger=None):
    funcs = [Gau_MVG_ML_estimates, Gau_Tied_ML_estimates, Gau_Naive_ML_estimates]
    models = ["MVG", "Tied MVG", "Naive Bayes MVG"]
    classes = ['Fake', 'Genuine']
    
    for m, f in zip(models, funcs):
        logger and logger.log_paragraph(f"{m} Pearson correlation")
        hParams = f(DTR, LTR)

        Cs = [ hParams[0][1], hParams[1][1] ]
        
        for i, label in enumerate(classes):
            logger.info(f'Class {label}')
            C = Cs[i]
            Corr = C / ( vcol(C.diagonal()**0.5) * vrow(C.diagonal()**0.5) )
            Corr_rounded = numpy.round(Corr, 3)
            logger.log(f"Pearson correlation:\n{Corr_rounded}")


def analyze_MVG_trunc_features(DTR, LTR, DVAL, LVAL, logger=None):
    cases = ["First 4 features", "Features 1-2", "Features 3-4"]
    DTRs = [DTR[:4], DTR[:2], DTR[2:4]]
    DVALs = [DVAL[:4], DVAL[:2], DVAL[2:4]]

    for case, DT, DV in zip(cases, DTRs, DVALs):
        logger and logger.log_paragraph(f"Trunc dataset: {case}")

        # MVG
        PVAL_MVG = binaryMVGModels("MVG", DT, LTR, DV, LVAL, ML_func=Gau_MVG_ML_estimates, logger=logger)

        # MVG Tied
        PVAL_Tied = binaryMVGModels("Tied MVG", DT, LTR, DV, LVAL, ML_func=Gau_Tied_ML_estimates, logger=logger)

        if case == cases[0]:
            # MVG Naive Bayes
            PVAL_Naive = binaryMVGModels("Naive Bayes MVG", DT, LTR, DV, LVAL, ML_func=Gau_Naive_ML_estimates, logger=logger)

def analyze_MVG_PCA(DTR, LTR, DVAL, LVAL, directions, logger=None, save_tables=None):
    PVALs = []
    
    for m in range(directions):
        logger and logger.log_paragraph(f'PCA pre-processing - {m+1} directions\n')
        DTR_pca, DVAL_pca = execute_PCA(DTR, m+1, logger, DVAL)

        # MVG
        PVAL_MVG = binaryMVGModels("MVG", DTR_pca, LTR, DVAL_pca, LVAL, ML_func=Gau_MVG_ML_estimates, logger=logger)

        # MVG Tied
        PVAL_Tied = binaryMVGModels("Tied MVG", DTR_pca, LTR, DVAL_pca, LVAL, ML_func=Gau_Tied_ML_estimates, logger=logger)

        # MVG Naive Bayes
        PVAL_Naive = binaryMVGModels("Naive Bayes MVG", DTR_pca, LTR, DVAL_pca, LVAL, ML_func=Gau_Naive_ML_estimates, logger=logger)

        PVALs.append([PVAL_MVG, PVAL_Tied, PVAL_Naive])

    if save_tables:
        header = ["PCA Directions", "MVG", "Tied MVG", "Naive Bayes MVG"]
        rows = [ [f"Direction {d+1}", *[format(x, ".2f") for x in compute_error_rates_multi(row, LVAL)]] for d, row in enumerate(PVALs) ]

        save_csv(rows, header, logger, "All_MVG_PCA", "L5_generative_models")
