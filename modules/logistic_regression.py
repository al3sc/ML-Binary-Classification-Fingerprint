import numpy
import scipy.special

import utils.plot as P

from .data_management import vrow, vcol, save_csv
from .dimensionality_reduction import execute_PCA
from .bayes_decisions_model_evaluation import compute_model_DCFs

# Optimize the logistic regression loss
def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = (vcol(w).T @ DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    # print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

# Optimize the weighted logistic regression loss
def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    #print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]


def update_logReg_DCF(LVAL, prior, scores, p, actDCFs, minDCFs, error_rates, TABLE, args, logger=None):
    predictions = (scores > 0) * 1      # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
    err = (predictions != LVAL).sum() / float(LVAL.size)
    error_rates.append(err*100)

    # DCF computing
    s_llr = scores - numpy.log(p / (1 - p))

    _, actDCF, minDCF = compute_model_DCFs(s_llr, prior, 1, 1, LVAL)
    # actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(s_llr, LVAL, prior)
    # minDCF = compute_minDCF_binary_fast(s_llr, LVAL, prior, returnThreshold=False)
    actDCFs.append(actDCF)
    minDCFs.append(minDCF)

    if logger:
        logger.info("TRAINING RESULTS")
        logger.log()
        logger.log(f"actDCF: {actDCF:.3} - minDCF: {minDCF:.3}")
        logger.log(f"Error Rate: {err*100}% - Accuracy: {(1-err)*100}%")
    
    if args.save_tables:
        TABLE.append(round(actDCF, 3))
        TABLE.append(round(minDCF, 3))
        TABLE.append(round(err*100, 2))
    
    return actDCFs, minDCFs, error_rates, TABLE

def log_train_results(actDCFs, minDCFs, error_rates, logger):
    logger.log_separator()
    logger.log("All actDCF obtained:")
    logger.log(actDCFs)
    logger.log("All minDCF obtained:")
    logger.log(minDCFs)
    logger.log("All errors obtained:")
    logger.log(error_rates)


def expand_features_quadratic(D):
	n_features, _ = D.shape
	expanded = [D]

	# Add squared terms
	expanded.append(D ** 2)

	# Add pairwise interaction terms
	for i in range(n_features):
		for j in range(i + 1, n_features):
			expanded.append(D[i, :] * D[j, :])

	return numpy.vstack(expanded)



def train_basic_LogReg(DTR, LTR, DVAL, LVAL, lambdas, prior, pEmp, args, logger=None, out_name=None):
    error_rates = []
    actDCFs = []
    minDCFs = []
    parameters = []

    TABLE = [ [l] for l in lambdas]
    HEADER = [ "λ", "actDCF", "minDCF", "ERROR RATE (%)"]


    if logger:
        logger.log(f"lambdas = {lambdas.shape}")
        logger.log(lambdas.tolist())


    for i, l in enumerate(lambdas):
        logger and logger.log_paragraph(f"{i+1}. Training Basic LogReg with λ = {l}")
        
        w, b = trainLogRegBinary(DTR, LTR, l)
        sVal = w.T @ DVAL + b # Compute validation scores
        
        actDCFs, minDCFs, error_rates, TABLE[i] = update_logReg_DCF(LVAL, prior, sVal, pEmp, actDCFs, minDCFs, error_rates, TABLE[i], args, logger)
        if args.save_models:
            parameters.append({'lambda': l, 'parameters': numpy.hstack([w, b])})
    
    # Plot results
    P.plot_log(
        lambdas, error_rates, title="Error rates - Basic Logistic Regression",
        xlabel=f"Regularization parameter λ", ylabel="Error rate",
        label0="Error rate",
        save_disk=args.save_plots, output_dir=f"{args.output}/L8_LogReg", output_name=f"ER_{out_name if out_name else "Basic"}_LogReg"
    )
    P.plot_log(
        lambdas, actDCFs, minDCFs, title=f"actDCF vs minDCF - {out_name if out_name else "Basic"} Logistic regression",
        xlabel=f"Regularization parameter λ", ylabel="DCF",
        label0="actDCF", label1="minDCF",
        save_disk=args.save_plots, output_dir=f"{args.output}/L8_LogReg", output_name=f"DCF_{out_name if out_name else "Basic"}_LogReg"
    )

    if logger:
        log_train_results(actDCFs, minDCFs, error_rates, logger)
        
        logger.log_separator()
        logger.log(f"Logistic Regression training ({out_name if out_name else "Basic"} version) completed!")

    if args.save_models:        
        numpy.save(f'{args.output}/L8_LogReg/models/{out_name if out_name else "Basic"}.npy', parameters)

    if args.save_tables:
        save_csv(TABLE, HEADER, logger, output_dir=f"{args.output}/L8_LogReg", output_file=f"LogReg_{out_name if out_name else "Basic"}")


def train_weighted_LogReg(DTR, LTR, DVAL, LVAL, lambdas, prior, priors, args, logger=None):
    error_rates = [[] for _ in priors]
    actDCFs = [[] for _ in priors]
    minDCFs = [[] for _ in priors]
    parameters = []

    TABLE = [ [p, l] for p in priors for l in lambdas]
    HEADER = [ "π", "λ", "actDCF", "minDCF", "ERROR RATE (%)"]


    if logger:
        logger.log(f"lambdas = {lambdas.shape}")
        logger.log(lambdas.tolist())


    for i, p in enumerate(priors):
	
        logger and logger.log_paragraph(f"{i+1}) Training Weighted LogReg with π = {p}")
        
        for j, l in enumerate(lambdas):
            logger and logger.log_paragraph(f"{j+1}. λ = {l}")
            
            w, b = trainWeightedLogRegBinary(DTR, LTR, l, p)
            sVal = w.T @ DVAL + b # Compute validation scores
            
            actDCFs[i], minDCFs[i], error_rates[i], TABLE[(i*len(lambdas))+j] = update_logReg_DCF(LVAL, prior, sVal, p, actDCFs[i], minDCFs[i], error_rates[i], TABLE[(i*len(lambdas))+j], args, logger)
            if args.save_models:
                parameters.append({'lambda': l, 'parameters': numpy.hstack([w, b])})

            logger and log_train_results(actDCFs[i], minDCFs[i], error_rates[i], logger)
     
        if args.save_models:        
            numpy.save(f'{args.output}/L8_LogReg/models/Weighted_P_0_{str(round(p,1)).replace('.','-')}.npy', parameters)
    
    # Plot results
    er_labels = [ f"Error rate (π = {p})" for p in priors]
    labels0 = [ f"actDCF (π = {p})" for p in priors]
    labels1 = [ f"minDCF (π = {p})" for p in priors]
    
    P.plot_log(
        lambdas, error_rates, title="Error rates - Weighted Logistic Regression",
        xlabel=f"Regularization parameter λ", ylabel="Error rate",
        label0=er_labels,
        save_disk=args.save_plots, output_dir=f"{args.output}/L8_LogReg", output_name=f"ER_Weighted_LogReg"
    )
    P.plot_log(
        lambdas, actDCFs, minDCFs, title=f"actDCF vs minDCF - Weighted Logistic regression",
        xlabel=f"Regularization parameter λ", ylabel="DCF",
        label0=labels0, label1=labels1,
        save_disk=args.save_plots, output_dir=f"{args.output}/L8_LogReg", output_name=f"DCF_Weighted_LogReg"
    )

    if logger:
        logger.log_separator()
        logger.log(f"Logistic Regression training (Weighted version) completed!")

    if args.save_tables:
        save_csv(TABLE, HEADER, logger, output_dir=f"{args.output}/L8_LogReg", output_file=f"LogReg_Weighted")


def train_PCA_LogReg(DTR, LTR, DVAL, LVAL, lambdas, prior, p, args, m=6, logger=None):
    error_rates = [[] for _ in range(m)]
    actDCFs = [[] for _ in range(m)]
    minDCFs = [[] for _ in range(m)]
    parameters = []

    TABLE = [ [m+1, l] for m in range(6) for l in lambdas]
    HEADER = [ "PCA directions (m)", "λ", "actDCF", "minDCF", "ERROR RATE (%)"]

    if logger:
        logger.log(f"lambdas = {lambdas.shape}")
        logger.log(lambdas.tolist())

    for i in range(m):
        DTR_pca, DVAL_pca = execute_PCA(DTR, i+1, None, DVAL)
	
        logger and logger.log_paragraph(f"{i+1}) Training LogReg with m = {i+1} PCA directions")
        
        for j, l in enumerate(lambdas):
            logger and logger.log_paragraph(f"{j+1}. λ = {l}")
            
            w, b = trainWeightedLogRegBinary(DTR_pca, LTR, l, p)
            sVal = w.T @ DVAL_pca + b # Compute validation scores
            
            actDCFs[i], minDCFs[i], error_rates[i], TABLE[(i*len(lambdas))+j] = update_logReg_DCF(LVAL, prior, sVal, p, actDCFs[i], minDCFs[i], error_rates[i], TABLE[(i*len(lambdas))+j], args, logger)
            if args.save_models:
                parameters.append({'lambda': l, 'parameters': numpy.hstack([w, b])})

            logger and log_train_results(actDCFs[i], minDCFs[i], error_rates[i], logger)    
     
        if args.save_models:        
            numpy.save(f'{args.output}/L8_LogReg/models/PCA_{i+1}.npy', parameters)
    
    # Plot results
    er_labels = [ f"Error rate (π = {p})" for p in range(m)]
    labels0 = [ f"actDCF (π = {p})" for p in range(m)]
    labels1 = [ f"minDCF (π = {p})" for p in range(m)]
    
    P.plot_log(
        lambdas, error_rates, title="Error rates - PCA Logistic Regression",
        xlabel=f"Regularization parameter λ", ylabel="Error rate",
        label0=er_labels,
        save_disk=args.save_plots, output_dir=f"{args.output}/L8_LogReg", output_name=f"ER_PCA_LogReg"
    )
    P.plot_log(
        lambdas, actDCFs, minDCFs, title=f"actDCF vs minDCF - PCA Logistic regression",
        xlabel=f"Regularization parameter λ", ylabel="DCF",
        label0=labels0, label1=labels1,
        save_disk=args.save_plots, output_dir=f"{args.output}/L8_LogReg", output_name=f"DCF_PCA_LogReg"
    )

    if logger:
        logger.log_separator()
        logger.log(f"Logistic Regression training (PCA version) completed!")

    if args.save_tables:
        save_csv(TABLE, HEADER, logger, output_dir=f"{args.output}/L8_LogReg", output_file=f"LogReg_PCA")
