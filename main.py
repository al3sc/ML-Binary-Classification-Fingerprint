from utils.logger import Logger
from utils.args_utils import arg_parse, validate_args, should_execute

from modules.data_management import load, split_db_2to1
from modules.data_visualization import visualize_data, compute_statistics
from modules.dimensionality_reduction import execute_PCA, visualize_data_PCA, execute_LDA, visualize_data_LDA, \
        execute_LDA_TrVal, classify_LDA, classify_LDA_manyThresholds, classify_LDA_prePCA
from modules.gaussian_density_estimation import fit_univariate_Gaussian_toFeatures
from modules.generative_models import apply_all_binary_MVG, analyze_C_MVG_models, compute_Pearson_correlation, \
        analyze_MVG_trunc_features, analyze_MVG_PCA
from modules.bayes_decisions_model_evaluation import analyze_applications, optimal_Bayes_decisions_effective_applications, \
        find_best_models_minDCF, find_best_calibrationLoss, visualize_Bayes_errors
from modules.GMM import train_evaluate_GMM


def main():
    ###################################################################################################
    # 1) INITIALIZATION

    args = arg_parse()
    validate_args(args)
    logger = None

    inputFile = './assets/input/trainData.txt'
    D, L = load(inputFile)

    # DTR and LTR are model training data and labels
    # DVAL and LVAL are validation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


    ###################################################################################################
    # 2) Data visualization

    if should_execute(2, args.modules):
        print("2 Data visualization...")

        # Initialize logger
        if args.log:
            logger = Logger("data_visualization", resume=args.resume)

        hFea = {
                0: 'Feature 1',
                1: 'Feature 2',
                2: 'Feature 3',
                3: 'Feature 4',
                4: 'Feature 5',
                5: 'Feature 6'
            }

        # PLOTS

        visualize_data(D, L, hFea, args)

        # DATASET and PER-CLASS STATISTICS

        logger and logger.log_title("Explore dataset statistics")
        compute_statistics(D, L, logger)

        if args.log:
            logger.__close__()


   
    ###################################################################################################
    # 3) Dimensionality Reduction

    if should_execute(3, args.modules):
        print("3 Dimensionality reduction...")

        # Initialize logger
        if args.log:
            logger = Logger("dimensionality_reduction", resume=args.resume)


        ### PCA - Principal Component Analysis

        logger and logger.log_title("Analyzing the effects of PCA on the features.")
        m = 6
        D_PCA, _ = execute_PCA(D, m, logger)

        # Plot the results of applying PCA on the features
        visualize_data_PCA(D_PCA, L, m, args)


        ### LDA - Linear Discriminant Analysis

        logger and logger.log_title("Analyzing the effects of LDA on the features.")
        
        D_LDA = execute_LDA(D, L, logger)

        # Plot the results of applying LDA on the features
        visualize_data_LDA(D_LDA, L, args)


        ### LDA for classification
        
        if args.log:
            logger.log_title("Applying LDA as a classifier.")
            logger.log_paragraph("Apply LDA to training and validation data.")
        
        DTR_LDA, DVAL_LDA = execute_LDA_TrVal(DTR, LTR, DVAL, logger)

        # Plot the results
        visualize_data_LDA(DTR_LDA, LTR, args, "LDA training set")
        visualize_data_LDA(DVAL_LDA, LVAL, args, "LDA validation set")

        # classification

        logger and logger.log_title("Perform classification task with LDA.")
        
        PVAL = classify_LDA(DTR_LDA, LTR, DVAL_LDA, LVAL, logger)

        # change thresholds for classification
        
        logger and logger.log_title("Perform classification exploring different thresholds.")
        PVALs = classify_LDA_manyThresholds(DTR, LTR, DVAL, LVAL, logger, save_tables=args.save_tables)


        # classification pre-processing the features with PCA

        logger and logger.log_title("Perform classification task - PCA for pre-processing.")
        
        PVALs = classify_LDA_prePCA(DTR, LTR, DVAL, LVAL, m, logger, save_tables=args.save_tables)        

        if args.log:
            logger.__close__()
    


    ###################################################################################################
    # 4) Gaussian Density Estimation

    if should_execute(4, args.modules):
        print("4 Gaussian Density Estimation...")

        # Initialize logger
        if args.log:
            logger = Logger("gaussian_density_estimation", resume=args.resume)

        fit_univariate_Gaussian_toFeatures(D, L, args.save_plots, logger)


        if args.log:
            logger.__close__()
    

    ###################################################################################################
    # 5) Generative models

    if should_execute(5, args.modules):
        print("5 Generative models...")

        # Initialize logger
        if args.log:
            logger = Logger("generative_models", resume=args.resume)

        # Apply MVG, Tied MVG and Naive Bayes MVG models
        logger and logger.log_title("MVG models application")
        apply_all_binary_MVG(DTR, LTR, DVAL, LVAL, args, logger)

        # Analyze results in light of characteristics
        logger and logger.log_title("MVG models results analysis")
        analyze_C_MVG_models(DTR, LTR, logger)

        # Compute Pearson correlation to better visualize the strength of covariances w.r.t. variances
        logger and logger.log_title("MVG models Pearson correlations")
        compute_Pearson_correlation(DTR, LTR, logger)

        # Trunc the dataset and analyze the MVG models w.r.t different features
        logger and logger.log_title("MVG models on different sets of features")
        analyze_MVG_trunc_features(DTR, LTR, DVAL, LVAL, logger)

        # Analyze the MVG models w.r.t PCA directions
        logger and logger.log_title("MVG models on different PCA directions")
        m = 6
        analyze_MVG_PCA(DTR, LTR, DVAL, LVAL, m, args, logger)


        if args.log:
            logger.__close__()
    


    ###################################################################################################
    # 7) Bayes Decisions Model Evaluation

    if should_execute(7, args.modules):
        print("7 Bayes Decisions Model Evaluation...")

        # Initialize logger
        if args.log:
            logger = Logger("bayes_decisions_model_evaluation", resume=args.resume)

        # Analyze error rates and confusion matrices of 5 applications 
        logger and logger.log_title("Considering the first 5 applications")
        applications = [
            (0.5, 1.0, 1.0),
            (0.9, 1.0, 1.0),
            (0.1, 1.0, 1.0),
            (0.5, 1.0, 9.0),
            (0.5, 9.0, 1.0)
        ]
        analyze_applications(DTR, LTR, DVAL, LVAL, applications, logger)
        
        # analyze applications with effective priors
        logger and logger.log_title("Considering the effective applications")
        effective_applications = [
            (0.5, 1.0, 1.0),
            (0.9, 1.0, 1.0),
            (0.1, 1.0, 1.0)
        ]
        DCFs = optimal_Bayes_decisions_effective_applications(DTR, LTR, DVAL, LVAL, effective_applications, args, logger)

        # find the best minDCF
        logger and logger.log_title("Best models in terms of minDCF")
        find_best_models_minDCF(DCFs, logger=logger)

        #find the best calibration loss
        logger and logger.log_title("Best models in terms of calibration loss")
        find_best_calibrationLoss(DCFs, logger=logger)

        # visualize Bayes error of the best PCA configuration
        visualize_Bayes_errors(DTR, LTR, DVAL, LVAL, DCFs, args)

        if args.log:
            logger.__close__()
    
    


    ###################################################################################################
    # 10) Gaussian Mixture Models

    if should_execute(10, args.modules):
        print("10 Gaussian Mixture Models...")

        # Initialize logger
        if args.log:
            logger = Logger("gaussian_mixture_models", resume=args.resume)

        nComponents = [ 2**i for i in range(6) ]
        train_evaluate_GMM(DTR, LTR, DVAL, LVAL, nComponents, args, logger )


        if args.log:
            logger.__close__()

    

    ###################################################################################################
    # 10.1) Best perfomring models

    if should_execute(10, args.modules):
        print("10.1 Best performing models...")

        # Initialize logger
        if args.log:
            logger = Logger("best_performing_models", resume=args.resume)

        

        if args.log:
            logger.__close__()
    


if __name__ == "__main__":
    
    main()