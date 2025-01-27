from utils.logger import Logger
from utils.args_utils import arg_parse, validate_args, should_execute

from modules.data_management import load, split_db_2to1
from modules.data_visualization import visualize_data, compute_statistics
from modules.dimensionality_reduction import execute_PCA, visualize_data_PCA, execute_LDA, visualize_data_LDA, \
        execute_LDA_TrVal, classify_LDA, classify_LDA_manyThresholds, classify_LDA_prePCA


def main():
    ###################################################################################################
    # 1) INITIALIZATION

    args = arg_parse()
    validate_args(args)

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

        if args.log:
            logger.log_title("Explore dataset statistics")
        compute_statistics(D, L, logger if args.log else None)

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

        if args.log:
            logger.log_title("Analyzing the effects of PCA on the features.")
        m = 6
        D_PCA, _ = execute_PCA(D, m, logger if args.log else None)

        # Plot the results of applying PCA on the features
        visualize_data_PCA(D_PCA, L, m, args)


        ### LDA - Linear Discriminant Analysis

        if args.log:
            logger.log_title("Analyzing the effects of LDA on the features.")
        
        D_LDA = execute_LDA(D, L, logger if args.log else None)

        # Plot the results of applying LDA on the features
        visualize_data_LDA(D_LDA, L, args)


        ### LDA for classification
        
        if args.log:
            logger.log_title("Applying LDA as a classifier.")
            logger.log_paragraph("Apply LDA to training and validation data.")
        
        DTR_LDA, DVAL_LDA = execute_LDA_TrVal(DTR, LTR, DVAL, logger if args.log else None)

        # Plot the results
        visualize_data_LDA(DTR_LDA, LTR, args, "LDA training set")
        visualize_data_LDA(DVAL_LDA, LVAL, args, "LDA validation set")

        # classification

        if args.log:
            logger.log_title("Perform classification task with LDA.")
        
        PVAL = classify_LDA(DTR_LDA, LTR, DVAL_LDA, LVAL, logger if args.log else None)

        if args.log:
            logger.log_title("Perform classification exploring different thresholds.")
        # change thresholds for classification
        PVALs = classify_LDA_manyThresholds(DTR, LTR, DVAL, LVAL, logger if args.log else None)

        # classification pre-processing the features with PCA

        if args.log:
            logger.log_title("Perform classification task - PCA for pre-processing.")
        
        PVALs = classify_LDA_prePCA(DTR, LTR, DVAL, LVAL, m, logger if args.log else None)

        

        if args.log:
            logger.__close__()
    


    ###################################################################################################
    # 4) Gaussian Density Estimation

    if should_execute(4, args.modules):
        print("4 Gaussian Density Estimation...")

        # Initialize logger
        if args.log:
            logger = Logger("gaussian_density_estimation", resume=args.resume)



        if args.log:
            logger.__close__()
    


if __name__ == "__main__":
    
    main()