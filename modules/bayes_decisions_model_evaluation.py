import numpy

import utils.plot as P

from .data_management import save_csv
from .dimensionality_reduction import execute_PCA
from .generative_models import compute_log_likelihood_Gau, Gau_MVG_ML_estimates, Gau_Tied_ML_estimates, Gau_Naive_ML_estimates

# Optimal Bayes deicsions for binary tasks with log-likelihood-ratio scores
def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

# Assume that classes are labeled 0, 1, 2 ... (nClasses - 1)
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

# Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)


# Compute minDCF (fast version)
# If we sort the scores, then, as we sweep the scores, we can have that at most one prediction changes everytime. We can then keep a running confusion matrix (or simply the number of false positives and false negatives) that is updated everytime we move the threshold

# Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
# We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs
    classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0 # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    #The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    #Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    #Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]: # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
          
    return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(thresholdsOut) # we return also the corresponding thresholds
    
def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = numpy.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]



def binaryMVG_models_llr(DTR, LTR, DVAL, ML_func=Gau_MVG_ML_estimates):
    hParams = ML_func(DTR, LTR)

    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams)
    llr = S_logLikelihood[1] - S_logLikelihood[0]

    return llr

def compute_model_DCFs(llr, prior, Cfn, Cfp, classLabels):
    predictions = compute_optimal_Bayes_binary_llr(llr, prior, 1, 1)
    DCFu = compute_empirical_Bayes_risk_binary(predictions, classLabels, prior, Cfn, Cfp, normalize=False)
    actDCF = compute_empirical_Bayes_risk_binary(predictions, classLabels, prior, Cfn, Cfp, normalize=True)
    minDCF = compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False)
    return DCFu, actDCF, minDCF

def train_model_DCF(DTR, LTR, DVAL, LVAL, prior, Cfn=1, Cfp=1, ML_func=Gau_MVG_ML_estimates):
    llr = binaryMVG_models_llr(DTR, LTR, DVAL, ML_func)
    return compute_model_DCFs(llr, prior, Cfn, Cfp, LVAL)


# Analyze first 5 applications

def computeConfusionMatrixMVG(DTR, LTR, DVAL, LVAL, ML_func=Gau_MVG_ML_estimates, prior1=1/2, Cfn=1.0, Cfp=1.0):
    hParams = ML_func(DTR, LTR)

    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams)

    llr = S_logLikelihood[1] - S_logLikelihood[0]
    threshold = -numpy.log( (prior1 * Cfn) / ((1 - prior1) * Cfp) )       # new threshold

    predictions_class1 = numpy.array(llr >= threshold)

    tp = numpy.sum((predictions_class1 == 1) & (LVAL == 1))  # True Positives
    tn = numpy.sum((predictions_class1 == 0) & (LVAL == 0))  # True Negatives
    fp = numpy.sum((predictions_class1 == 1) & (LVAL == 0))  # False Positives
    fn = numpy.sum((predictions_class1 == 0) & (LVAL == 1))  # False Negatives

    # Confusion matrix as 2x2 array
    confusion_matrix = numpy.array([[tp, fn], [fp, tn]])

    return confusion_matrix

def analyze_applications(DTR, LTR, DVAL, LVAL, applications, logger=None):
    for i, t in enumerate(applications):
        logger and logger.log_paragraph(f"MVG on Application {i+1}")
        
        prior_T = t[0]
        Cfn = t[1]
        Cfp = t[2]

        if logger:
            logger.log(f"π_T = {prior_T}, π_F = {1-prior_T:.1f}\nCfn = {Cfn}, Cfp = {Cfp}\n")
        
        effective_prior = (prior_T * Cfn) / ( (prior_T * Cfn) + ((1 - prior_T) * Cfp) )
        if effective_prior != prior_T and logger:
            logger.log(f"Effective prior: {effective_prior}")
            logger.log(f"Equivalent application: (π_T={effective_prior}, Cfn=1, Cfp=1)")

        confusion_matrix = computeConfusionMatrixMVG(DTR, LTR, DVAL, LVAL, prior1=prior_T, Cfn=Cfn, Cfp=Cfp)
        if logger:
            logger.log("Confusion Matrix:")
            logger.log(f"{confusion_matrix}\n")


def getResults(DCFs):
    models = ["MVG", "Tied MVG", "Naive Bayes MVG"]
    results = []
    for app_prior, modes in DCFs.items():
        for mode, models in modes.items():
            for model, dcfs in models.items():
                results.append((app_prior, mode, model, dcfs[0], dcfs[1]))
    return results

def optimal_Bayes_decisions_effective_applications(DTR, LTR, DVAL, LVAL, applications, args, logger=None):
    DCFs = {}
    models = {'MVG': Gau_MVG_ML_estimates, 'Tied MVG': Gau_Tied_ML_estimates , 'Naive Bayes MVG':Gau_Naive_ML_estimates}

    for i, t in enumerate(applications):
        logger and logger.log_paragraph(f"Application {i+1}")
    
        prior = t[0]
        Cfn = t[1]
        Cfp = t[2]

        if logger:
            logger.log(f"π_T = {prior}, π_F = {1-prior:.1f}\nCfn = {Cfn}, Cfp = {Cfp}")

        DCFs[t[0]] = {}
        
        for m in range(6+1):
            if m!=0:
                logger and logger.log_paragraph(f'PCA pre-processing - {m} directions\n')
                DTR_pca, DVAL_pca = execute_PCA(DTR, m, None, DVAL)
                DTR_selected = DTR_pca
                DVAL_selected = DVAL_pca

            else:
                logger and logger.log_paragraph(f'No PCA pre-processing')
                DTR_selected = DTR
                DVAL_selected = DVAL

            dict_iter_name = f"PCA - {m} directions" if m!=0 else "No PCA"
            DCFs[t[0]][dict_iter_name] = {}

            for model, func in models.items():
                DCFu, actDCF, minDCF = train_model_DCF(DTR_selected, LTR, DVAL_selected, LVAL, prior, ML_func = func)
                DCFs[t[0]][dict_iter_name][model] = [actDCF, minDCF]
            
                logger and logger.log(f"{model:<20}Non-normalized DCF: {DCFu:.3f} - Normalized DCF: {actDCF:.3f} - minDCF: {minDCF:.3f}.")

    if args.save_tables:
        header = ["App prior", "Mode", "Model", "actDCF", "minDCF"]
        rows = getResults(DCFs)
        save_csv(rows, header, logger, output_file=f"DCFs_MVG", output_dir=f"{args.output}/L7_bayes_decisions_model_evaluation")
    
    return DCFs


def find_best_models_minDCF(DCFs, n=None, logger=None):
    results = getResults(DCFs)
    
    results.sort(key=lambda x: x[4])
    top_models = results[:n] if n else results

    if logger:
        logger.log(f"{'#':<5} {'Prior':<10} {'Mode':<20} {'Model':<20} {'minDCF':<10}")
        logger.log("=" * 70)

        for i, model in enumerate(top_models):
            logger.log(f"{i+1:<5} {model[0]:<10} {model[1]:<20} {model[2]:<20} {model[4]:<10.3f}")

def find_best_calibrationLoss(DCFs, bound_percentage=10, logger=None):
    results = getResults(DCFs)

    def getCalibrationLoss(actDCF, minDCF):
        return round( (actDCF - minDCF) / minDCF * 100, 2 )

    results.sort(key=lambda x: getCalibrationLoss(x[3], x[4]))
    top_models = [x for x in results if getCalibrationLoss(x[3], x[4]) <= bound_percentage]

    if logger:
        logger.log(f"{'#':<5} {'Prior':<10} {'Mode':<20} {'Model':<20} {'actDCF':<10} {'minDCF':<10} {'Calibration Loss':<15}")
        logger.log("=" * 95)

        for i, model in enumerate(top_models):
            logger.log(f"{i+1:<5} {model[0]:<10} {model[1]:<20} {str(model[2]):<20} {model[3]:<10.3f} {model[4]:<10.3f} {getCalibrationLoss(model[3], model[4]):<15.2f}")


def visualize_Bayes_errors(DTR, LTR, DVAL, LVAL, DCFs, args):
    models = {'MVG': Gau_MVG_ML_estimates, 'Tied MVG': Gau_Tied_ML_estimates , 'Naive Bayes MVG':Gau_Naive_ML_estimates}
    
    #our_models = DCFs[(0.1, 1.0, 1.0)]
    #DCFs = DCFs[0.1]
    results = getResults(DCFs)
    results = [row for row in results if row[0] == 0.1 and row[1] != "No PCA"]         # choose only the application with prior=0.1
    best_PCA_model = sorted(results, key=lambda x: x[4])[0]                     # choose the PCA configuration with the lower minDCF
    m = int(best_PCA_model[1].split()[2])
    DTR_pca, DVAL_pca = execute_PCA(DTR, m, None, DVAL)

    #mvg = best_PCA_model['MVG']

    effPriorLogOdds = numpy.linspace(-4, 4, 30)
    actDCFs = {m: [] for m, _ in models.items()}
    minDCFs = {m: [] for m, _ in models.items()}

    for logOdd in effPriorLogOdds:
        prior = 1/(1 + numpy.exp(-logOdd))
        for model, func in models.items():
            _, actDCF, minDCF = train_model_DCF(DTR_pca, LTR, DVAL_pca, LVAL, prior, ML_func = func)
            actDCFs[model].append(actDCF)
            minDCFs[model].append(minDCF)


    P.plot_Bayes_error(effPriorLogOdds, actDCFs, minDCFs, models.keys(),
        title="Bayes error",
        save_disk = args.save_plots, output_dir=f"{args.output}/L7_bayes_decisions_model_evaluation", output_name="Bayes_error"
    )