import utils.project_module as pm
import utils.plot as P
from utils.logger import Logger

from modules.data_visualization import visualize_data, compute_statistics, compute_class_statistics


def main(logIsActive, save_plots):
    ###################################################################################################
    # 1) INITIALIZATION

    inputFile = './assets/input/trainData.txt'

    ###################################################################################################
    # 2) Data visualization

    # Initialize logger
    if logIsActive:
        logger = Logger("Data_Visualization")

    D, L = pm.load(inputFile)

    hFea = {
            0: 'Feature 1',
            1: 'Feature 2',
            2: 'Feature 3',
            3: 'Feature 4',
            4: 'Feature 5',
            5: 'Feature 6'
        }

    # PLOTS

    visualize_data(D, L, hFea, save_plots)

    # DATASET and PER-CLASS STATISTICS

    compute_statistics(D, L, logger if logIsActive else None)

    if logIsActive:
        logger.__close__()


    ###################################################################################################
    # 3) Dimensionality Reduction

    # Initialize logger
    if logIsActive:
        logger = Logger("Dimensionality_reduction")

    
    if logIsActive:
        logger.__close()



if __name__ == "__main__":
    logIsActive = False
    save_plots = False
    
    main(logIsActive, save_plots)