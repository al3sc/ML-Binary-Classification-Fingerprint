import numpy
import scipy.special

import utils.plot as P

from .data_management import vcol, vrow, save_csv
from .bayes_decisions_model_evaluation import compute_empirical_Bayes_risk_binary_llr_optimal_decisions, compute_minDCF_binary_fast


### FUNCTIONS

# Optimize Linear SVM
def train_dual_SVM_linear(DTR, LTR, C, K = 1, logger=None):
    
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])			# x, K -> x'
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)			# H' (H'_ij = z_i * z_j * x'_i.T * x'_j)

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()			# L'^D(alpha) = - J'^D(alpha) = 1/2 * alpha.T * H' * alpha - alpha.T * 1
        grad = Ha.ravel() - numpy.ones(alpha.size)							# gradient of the loss
        return loss, grad

	# position of the minimum, value of the func at the minimum, information dictionary
    alphaStar, _, d = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for _ in LTR], factr = 1.0, maxiter = 15000*1000)
    
    # Primal loss
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()                                                  # w'Star.T * x'_i
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()      # J' = 1/2 * ||w'Star||^2 + C * sum(max( 0, 1 - z_i*(w'Star.T * x'_i) ))

	# Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)                                # w'Star = sum( alphaStar * z_i * x'_i )
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0].item()
    
    if logger:
        logger.log("L_BFGS_B info", lvl="INFO")
        logger.log(f"Minimum position: {alphaStar}")
        logger.log(f"Number of fun calls: {d['funcalls']}")
        logger.log()
        logger.log("Training info", lvl="INFO")
        logger.log(f"w_hat: {w_hat}")
        logger.log('primal loss %e - dual loss %e - duality gap %e' % (primalLoss, dualLoss, primalLoss - dualLoss))
        logger.log()
    
    return w, b, primalLoss, dualLoss


### NON-LINEAR SVM FUNCTIONS

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc


# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0, logger=None):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps                              # k'(x_1, x_2) = k(x_1, x_2) + eps      (eps = non-linear SVM regularized bias)
    H = vcol(ZTR) * vrow(ZTR) * K                         # H' (H'_ij = z_i * z_j * k(x_i.T * x_j))

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, d = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for _ in LTR], factr = 1.0, maxiter = 15000*1000)
    
    if logger:
        logger.log("L_BFGS_B info", lvl="INFO")
        logger.log(f"Minimum position: {alphaStar}")
        logger.log(f"Number of fun calls: {d['funcalls']}")
        logger.log()
        logger.log("Training info", lvl="INFO")
        logger.log('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0].item()))
        logger.log()

    # With non-linear SVM we are not able to compute the primal solution

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K                   # s(x_t)_i = alphaStar_i * z_i * k(x_i, x_t)
        return H.sum(0)                                             # s(x_t) = sum( s(x_t)_i )

    return fScore, alphaStar # we directly return the function to score a matrix of test samples



def update_local_DCF(LVAL, prior, scores, actDCFs, minDCFs, error_rates, TABLE, args, logger=None):
    actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(scores.ravel(), LVAL, prior, 1, 1, True)
    minDCF = compute_minDCF_binary_fast(scores.ravel(), LVAL, prior, 1, 1, returnThreshold=False)
    actDCFs.append(actDCF)
    minDCFs.append(minDCF)

    predictions = (scores > 0) * 1
    err = (predictions != LVAL).sum() / float(LVAL.size)
    error_rates.append(err*100)

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

def log_train_results(actDCFs, minDCFs, error_rates, logger, losses=None):
    logger.log_separator()
    logger.log("All actDCF obtained:")
    logger.log(actDCFs)
    logger.log("All minDCF obtained:")
    logger.log(minDCFs)
    logger.log("All errors obtained:")
    logger.log(error_rates)
    if losses:
        logger.log('All losses obtained')
        logger.log(losses)



def train_linear_SVM(DTR, LTR, DVAL, LVAL, C, args, prior=0.1, K=1, logger=None, out_name="original_data"):
    # Training Linear SVM

    if logger:
        logger.log(f"K = 1")
        logger.log(f"C = {C.shape}")
        logger.log(C.tolist())

    losses = []
    actDCFs = []
    minDCFs = []
    error_rates = []
    parameters = []

    # Table to be saved in a csv file if save_tables is true
    TABLE = [ [K, c] for c in C]
    HEADER = [ "K", "C", "actDCF", "minDCF", "ERROR RATE (%)"]

    for i, c in enumerate(C):
        logger and logger.log_paragraph(f"{i+1}. Training Linear SVM with C = {c:.3}")
        
        w, b, primalLoss, dualLoss = train_dual_SVM_linear(DTR, LTR, c, K)
        losses.append((primalLoss, dualLoss))

        scores = vrow(w) @ DVAL + b
        
        actDCFs, minDCFs, error_rates, TABLE[i] = update_local_DCF(LVAL, prior, scores, actDCFs, minDCFs, error_rates, TABLE[i], args, logger)
        
        if args.save_models:
            parameters.append({'C': c, 'parameters': numpy.hstack([w, b])})


    if args.save_models:        
        numpy.save(f'{args.output}/L9_SVM/models/L-SVM-{out_name}.npy', parameters)
    

    # Plot results
    P.plot_log(
        C, actDCFs, minDCFs, f"actDCF vs minDCF - Linear SVM - {out_name}",
        "C", "DCF", "actDCF", "minDCF",
        save_disk=args.save_plots, output_dir=f"{args.output}/L9_SVM", output_name=f"DCF_L_SVM-{out_name}")
        
    if logger:
        log_train_results(actDCFs, minDCFs, error_rates, logger, losses)

        logger.log_separator()
        logger.log(f"Linear SVM ({out_name}) training completed!")

    if args.save_tables:
        save_csv(TABLE, HEADER, logger, f"L_SVM-{out_name}", f"{args.output}/L9_SVM")


def train_polynomial_SVM(DTR, LTR, DVAL, LVAL, C, degree, args, prior=0.1, eps=0, c_kernel=1, logger=None):
    kernel = polyKernel(degree, c_kernel)

    if logger:
        logger.log(f"C = {C.shape}")
        logger.log(C.tolist())
        logger.log(f"ξ: {eps} - Kernel degree: {degree} - C (kernel): {c_kernel}")

    actDCFs = []
    minDCFs = []
    error_rates = []
    parameters = []

    # Table to be saved in a csv file if save_tables is true
    TABLE = [ [eps, degree, c] for c in C]
    HEADER = [ "ξ", "d", "C", "actDCF", "minDCF", "ERROR RATE (%)"]

    for i, c in enumerate(C):
        logger and logger.log_paragraph(f"{i+1}. Training N-L SVM with C = {c:.3}")
        
        fScore, alphaStar = train_dual_SVM_kernel(DTR, LTR, c, kernel, eps = eps)
        scores = fScore(DVAL)
        
        actDCFs, minDCFs, error_rates, TABLE[i] = update_local_DCF(LVAL, prior, scores, actDCFs, minDCFs, error_rates, TABLE[i], args, logger)
        
        if args.save_models:
            parameters.append({'C': c, 'parameters': alphaStar})


    if args.save_models:        
        numpy.save(f'{args.output}/L9_SVM/models/NL-SVM-poly_d{degree}.npy', parameters)
    

    # Plot results
    P.plot_log(
        C, actDCFs, minDCFs, f"actDCF vs minDCF - Non-Linear SVM - Polynomial kernel, degree = {degree}",
        "C", "DCF", "actDCF", "minDCF",
        save_disk=args.save_plots, output_dir=f"{args.output}/L9_SVM", output_name=f"DCF_NL_SVM-poly_d{degree}")
        
    if logger:
        log_train_results(actDCFs, minDCFs, error_rates, logger)

        logger.log_separator()
        logger.log(f"Non-linear SVM (polynomial degree = {degree}) training completed!")

    if args.save_tables:
        save_csv(TABLE, HEADER, logger, f"NL_SVM-poly_d{degree}", f"{args.output}/L9_SVM")


def train_RBF_SVM(DTR, LTR, DVAL, LVAL, C, gammas, args, prior=0.1, eps=1, logger=None):
    kernels = [rbfKernel(gamma) for gamma in gammas]
    k_names = [f"RBF Kernel with γ = {gamma:.3}" for gamma in gammas]

    if logger:
        logger.log(f"C = {C.shape}")
        logger.log(C.tolist())
        logger.log(f"ξ: {eps}")
        logger.log(f"Kernels: {k_names}")

    actDCFs = [[] for _ in range(len(kernels))]
    minDCFs = [[] for _ in range(len(kernels))]
    error_rates = [[] for _ in range(len(kernels))]
    parameters = []

    # Table to be saved in a csv file if save_tables is true
    gammas_strings = [f"exp(-{n+1})" for n in range(4)]
    TABLE = [ [eps, gamma, c] for gamma in gammas_strings for c in C]
    HEADER = [ "ξ", "γ", "C", "actDCF", "minDCF", "ERROR RATE (%)" ]

    for i, kernel in enumerate(kernels):
        logger and logger.log_paragraph(f"{i+1}) {k_names[i]}")
        
        for j, c in enumerate(C):
            logger and logger.log_paragraph(f"{j+1}. Training N-L SVM with C = {c:.3}")
            
            fScore, alphaStar = train_dual_SVM_kernel(DTR, LTR, c, kernel, eps = eps)
            scores = fScore(DVAL)
            
            actDCFs[i], minDCFs[i], error_rates[i], TABLE[(i*len(C))+j] = update_local_DCF(LVAL, prior, scores, actDCFs[i], minDCFs[i], error_rates[i], TABLE[(i*len(C))+j], args, logger)
            
            if args.save_models:
                parameters.append({'C': c, 'parameters': alphaStar})


        if args.save_models:        
            numpy.save(f'{args.output}/L9_SVM/models/NL-SVM-RBF_exp(-{i}).npy', parameters)
    
        logger and log_train_results(actDCFs[i], minDCFs[i], error_rates[i], logger)

    # Plot results
    labels0 = [ f"actDCF (γ = {gamma})" for gamma in gammas_strings]
    labels1 = [ f"minDCF (γ = {gamma})" for gamma in gammas_strings]
    P.plot_log(
        C, actDCFs, minDCFs, f"actDCF vs minDCF - Non-Linear SVM - RBF kernel",
        "C", "DCF", labels0, labels1,
        save_disk=args.save_plots, output_dir=f"{args.output}/L9_SVM", output_name=f"DCF_NL_SVM-RBF")
        
    if logger:
        logger.log_separator()
        logger.log(f"Non-linear SVM (RBF) training completed!")

    if args.save_tables:
        save_csv(TABLE, HEADER, logger, f"NL_SVM-RBF", f"{args.output}/L9_SVM")
