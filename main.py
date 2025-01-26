from utils.logger import Logger
from utils.args_utils import arg_parse, validate_args, should_execute

from modules.data_management import load
from modules.data_visualization import visualize_data, compute_statistics
from modules.dimensionality_reduction import execute_PCA, visualize_data_PCA, execute_LDA, visualize_data_LDA


def main():
    ###################################################################################################
    # 1) INITIALIZATION

    args = arg_parse()
    validate_args(args)

    inputFile = './assets/input/trainData.txt'
    D, L = load(inputFile)

    ###################################################################################################
    # 2) Data visualization

    if should_execute(2, args.modules):
        print("2 Data visualization...")

        # Initialize logger
        if args.log:
            logger = Logger("Data_Visualization")

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
            logger = Logger("Dimensionality_reduction")


        ### PCA - Principal Component Analysis

        if args.log:
            logger.log_title("Analyzing the effects of PCA on the features.")
        m = 6
        D_PCA = execute_PCA(D, m, logger if args.log else None)

        # Plot the results of applying PCA on the features
        visualize_data_PCA(D_PCA, L, m, args)


        ### LDA - Linear Discriminant Analysis

        if args.log:
            logger.log_title("Analyzing the effects of LDA on the features.")
        
        D_LDA = execute_LDA(D, L, logger if args.log else None)

        # Plot the results of applying LDA on the features
        visualize_data_LDA(D_LDA, L, args)

        
        if args.log:
            logger.__close()


        ### PCA & LDA for classification
        


if __name__ == "__main__":
    
    main()