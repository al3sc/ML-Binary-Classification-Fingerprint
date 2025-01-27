import argparse

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
        "--resume",
        action="store_true",
        help="Resume the logger file, when it already exists.",
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
    parser.add_argument(
        "--output",
        type=str,
        default="./assets/outputs",
        help="Output disk directory where data will be saved, if argument is set."
    )
    return parser.parse_args()

# Function to check if the argument are correct
def validate_args(args):
    # modules check
    modules = args.modules.split(",")
    valid_modules = [opt for sublist in MODULES.values() for opt in sublist]
    
    for m in modules:
        if m not in valid_modules:
            raise ValueError(f"Invalid module {m} specified. Valid options: {', '.join(valid_modules)}.")

# Function to check if a specific module should run
def should_execute(module_number, modules):
    if not modules:
        return True
    selected_modules = modules.split(",")
    return any(m in MODULES[module_number] for m in selected_modules)