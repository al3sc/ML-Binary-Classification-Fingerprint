import argparse

import utils.project_module as pm
import utils.plot as P
from utils.logger import Logger

from modules.data_management import load
from modules.data_visualization import visualize_data, compute_statistics, compute_class_statistics


MODULES = {
    2: ["2", "data_visualization", "DV"],
    3: ["3", "dimensionality_reduction", "DR"],
    4: ["4", "gaussian_density_estimation", "GDE"],
    5: ["5", "generative_models", "GM"],
    7: ["7", "bayes_decisions_model_evaluation", "BDME"],
    8: ["8", "logistic_regression", "LR"],
    9: ["9", "support_vector_machines", "SVM"],
    10: ["10", "gaussian_mixture_models", "GMM"],
    11: ["11", "calibration_fusion", "CF"],
}

def arg_parse():
    parser = argparse.ArgumentParser(description="Binary classification on Fingerprints - choose the section to execute.")

    parser.add_argument(
        "--modules",
        type=str,
        help=(
            "List of modules to execute, separated by a comma ',' ."
            "Numebrs can be used (ie: '1,2') or strings (ie: 'data_visualization,dimensionality_reduction')."
        ),
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Active the logger for logging the computational steps in a specific log file.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save plots on the disk.",
    )
    parser.add_argument(
        "--save_tables",
        action="store_true",
        help="Save the tables of the results on the disk.",
    )
    return parser.parse_args()

# Function to check if a specific module should run
def should_execute(module_number, modules):
    if modules:
        selected_modules = modules.split(",")
        if any(m in MODULES[module_number] for m in selected_modules):
        #if selected_modules in MODULES[module_number]:
            return True
        return False
    
    return True 


def main():
    ###################################################################################################
    # 1) INITIALIZATION

    args = arg_parse()

    inputFile = './assets/input/trainData.txt'

    ###################################################################################################
    # 2) Data visualization

    if should_execute(2, args.modules):
        print("2 Data visualization...")

        # Initialize logger
        if args.log:
            logger = Logger("Data_Visualization")

        D, L = load(inputFile)

        hFea = {
                0: 'Feature 1',
                1: 'Feature 2',
                2: 'Feature 3',
                3: 'Feature 4',
                4: 'Feature 5',
                5: 'Feature 6'
            }

        # PLOTS

        visualize_data(D, L, hFea, args.save_plots)

        # DATASET and PER-CLASS STATISTICS

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

        print("module 3")
        
        if args.log:
            logger.__close()



if __name__ == "__main__":
    
    main()