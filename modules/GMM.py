import numpy
import scipy.special

from .data_visualization import compute_mu_C
from .data_management import vcol, vrow, save_csv
from .gaussian_density_estimation import logpdf_GAU_ND
from .bayes_decisions_model_evaluation import compute_empirical_Bayes_risk_binary, compute_minDCF_binary_fast, compute_model_DCFs

import utils.plot as P

def logpdf_GMM(X, gmm):

    S = []                          # matrix (M, N), S[g, i] = logarithm of the joint probability of x_i and cluster g
    
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)            # log-marginal for all samples x_i, f_(X_i)(x_i)
    return logdens                                          # (N,), log-densities for each sample x_i


def smooth_covariance_matrix(C, psi):

    U, s, Vh = numpy.linalg.svd(C)
    s[s<psi]=psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd

def split_GMM_LBG(gmm, alpha = 0.1, logger=None):

    gmmOut = []
    if logger:
        logger.log('LBG - going from %d to %d components' % (len(gmm), len(gmm)*2))
    for (w, mu, C) in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut


# X: Data matrix
# gmm: input gmm
# covType: 'Full' | 'Diagonal' | 'Tied'
# psiEig: factor for eignvalue thresholding
#
# return: updated gmm
def train_GMM_EM_Iteration(X, gmm, covType = 'Full', psiEig = None): 

    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    
    # E-step
    S = []
    
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S) # Compute joint densities f(x_i, c), i=1...n, c=1...G
    logdens = scipy.special.logsumexp(S, axis=0) # Compute marginal for samples f(x_i)

    # Compute posterior for all clusters - log P(C=c|X=x_i) = log f(x_i, c) - log f(x_i)) - i=1...n, c=1...G
    # Each row for gammaAllComponents corresponds to a Gaussian component
    # Each column corresponds to a sample (similar to the matrix of class posterior probabilities in Lab 5, but here the rows are associated to clusters rather than to classes
    gammaAllComponents = numpy.exp(S - logdens)         # responsibilities

    # M-step
    gmmUpd = []
    for gIdx in range(len(gmm)): 
    # Compute statistics:
        gamma = gammaAllComponents[gIdx] # Extract the responsibilities for component gIdx
        Z = gamma.sum()                         # Zero order statistic
        F = vcol((vrow(gamma) * X).sum(1))      # First order statistic, Exploit broadcasting to compute the sum
        S = (vrow(gamma) * X) @ X.T             # Second order statistic
        muUpd = F/Z
        CUpd = S/Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd  = CUpd * numpy.eye(X.shape[0]) # An efficient implementation would store and employ only the diagonal terms, but is out of the scope of this script
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType.lower() == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]
        
    return gmmUpd

# Train a GMM until the average dela log-likelihood becomes <= epsLLAverage
def train_GMM_EM(X, gmm, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, logger=None):

    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    if logger:
        logger.log('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType = covType, psiEig = psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if logger:
            logger.log('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1

    if logger:
        logger.log('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))        
    return gmm
 

# Train a full model using LBG + EM, starting from a single Gaussian model, until we have numComponents components. lbgAlpha is the value 'alpha' used for LBG, the otehr parameters are the same as in the EM functions above
def train_GMM_LBG_EM(X, numComponents, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, lbgAlpha = 0.1, logger=None):

    mu, C = compute_mu_C(X)

    if covType.lower() == 'diagonal':
        C = C * numpy.eye(X.shape[0]) # We need an initial diagonal GMM to train a diagonal GMM
    
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))] # 1-component model - if we impose the eignevalus constraint, we must do it for the initial 1-component GMM as well
    else:
        gmm = [(1.0, mu, C)] # 1-component model
    
    while len(gmm) < numComponents:
        # Split the components
        if logger:
            logger.log('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha, logger=logger)
        if logger:
            logger.log('Average ll after LBG: %.8e' % logpdf_GMM(X, gmm).mean()) # NOTE: just after LBG the ll CAN be lower than before the LBG - LBG does not optimize the ll, it just increases the number of components
        # Run the EM for the new GMM
        gmm = train_GMM_EM(X, gmm, covType = covType, psiEig = psiEig, logger=logger, epsLLAverage = epsLLAverage)
    return gmm


def train_evaluate_GMM(DTR, LTR, DVAL, LVAL, nComponents, args, logger=None):
    
    prior, Cfn, Cfp = 0.1, 1, 1
    results = []
    for covType in ['full', 'diagonal', 'tied']:
        logger and logger.log_title(f"Training {covType} covariance GMM")

        best_results = {"minDCF": [float("inf"), 0, 0], "err": [float("inf"), 0, 0]}

        for nC_0 in nComponents:
            for nC_1 in nComponents:
                logger and logger.log_paragraph(f"Number of components: {nC_0} (class 0), {nC_1} (class 1)")
                logger and logger.log_paragraph(f"Training GMM for class 0")
                gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], nC_0, covType=covType, psiEig=1e-6, logger=logger)
                logger and logger.log_paragraph(f"Training GMM for class 1")
                gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], nC_1, covType=covType, psiEig=1e-6, logger=logger)
                
                S0 = logpdf_GMM(DVAL, gmm0) + numpy.log(prior)
                S1 = logpdf_GMM(DVAL, gmm1) + numpy.log(1 - prior)
                if logger:
                    logger.log_paragraph("Compute scores for each class")
                    logger.log(f"S0:\nSize: {S0.shape}\n{S0}")
                    logger.log(f"S1:\nSize: {S1.shape}\n{S1}")
                
                SLLR = S1 - S0

                SVAL = (SLLR > 0) * 1
                acc, err = numpy.mean(SVAL == LVAL), 1 - numpy.mean(SVAL == LVAL)
                logger and logger.log(f"Error rate (nC_0: {nC_0}, nC_1: {nC_1}): {err*100}%")
                _, actDCF, minDCF = compute_model_DCFs(SLLR, prior, Cfn, Cfp, LVAL)

                # best minDCF
                if minDCF < best_results["minDCF"][0]:
                    best_results["minDCF"] = [minDCF, nC_0, nC_1]

                # best error rate
                if err < best_results["err"][0]:
                    best_results["err"] = [err, nC_0, nC_1]

                if logger:
                    logger.log_paragraph(f"Computing DCF (nC_0: {nC_0}, nC_1: {nC_1})")
                    logger.log(f"actDCF: {actDCF}")
                    logger.log(f"minDCF: {minDCF}")
                
                if args.save_tables:
                    header = ["Cov Type", "n components 0", "n components 1", "error rate", "actDCF", "minDCF"]
                    results.append((covType, nC_0, nC_1, err*100, actDCF, minDCF))    
                    save_csv(results, header, logger, "GMM_LBG_EM", f"{args.output}/L10_GMM")

        if args.save_tables:
            header = ["minDCF", "mD_nC_0", "mD_nC_1", "error rate", "er_nC_0", "er_nC_1"]
            minDCF_best = best_results["minDCF"]
            err_best = best_results["err"]
            results = [round(minDCF_best[0], 3), minDCF_best[1], minDCF_best[2], round(err_best[0]*100, 3), err_best[1], err_best[2] ]
            
            save_csv(results, header, logger, f"best_results_GMM_{covType}", f"{args.output}/L10_GMM")