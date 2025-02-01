import os
import csv

import numpy


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                label = int(line.split(',')[-1].strip())
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def split_classes(D, L, n_classes):
    D_split = []
    for cls in range(n_classes):
        D_split.append(D[:, L==cls])
    return D_split

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def reduce_db(D, L, fraction = 0.5, seed = 0):
    n = int(D.shape[1] * fraction)
    if seed is not None:
        # choose samples randomly
        numpy.random.seed(seed)
        idx = numpy.random.permutation(D.shape[1])
        idxReduced = idx[0:n]
        D_reduced = D[:, idxReduced]
        L_reduced = L[idxReduced]
    else:
        # choose the first n samples of the dataset
        D_reduced = D[:, 0:n]
        L_reduced = L[0:n]
    return (D_reduced, L_reduced)

def save_csv(table, header, logger, output_file, output_dir="./assets/outputs"):
    """
    Save the table in a csv file.
    
    Args:
        table (list of lists): A 2D list where each inner list represents a row of data to be written.
        header (list): First row of the table.
        logger (object): Logger to log messages.
        output_file (str): The name of the output CSV file (without extension).
        output_dir (str, optional): The directory where the output file will be saved. Default is "./outputs".
    """
    if isinstance(header[0], list):  
        n = len(header[0])
    else:  
        n = len(header)
    
    for r in table:
        if len(r) < n:
            logger and logger.error("Some elements are missing in the elements of the tables!")
            raise ValueError("Some elements are missing in the elements of the tables!")
        elif len(r) > n:
            logger and logger.error("Some values are missing in the header!")
            raise ValueError("Some values are missing in the header!")
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = f"{output_dir}/{output_file}{"" if output_file.endswith(".csv") else ".csv"}"
        
    with open(file_path, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        if isinstance(header[0], list):  
            for h in header:
                writer.writerow(h)
        else:
            writer.writerow(header)

        for row in table:
            writer.writerow(row)
    
    if logger:
        logger.info(f"File '{output_file}' saved correctly!")
    else:
        print(f"File '{output_file}' saved correctly!")


def center_data(DTR, DVAL=None):
    mean = vcol(DTR.mean(1))
    DTR_centered = DTR - mean
    if DVAL is not None:
        DVAL_centered = DVAL - mean

        return DTR_centered, DVAL_centered
    return DTR_centered

def Z_normalization(DTR, DVAL):
	std = vcol(DTR.std(1))
	DTR_normalized = DTR / std
	DVAL_normalized = DVAL / std

	return DTR_normalized, DVAL_normalized


def compute_error_rate(PVAL, LVAL):
    nErrors = (PVAL != LVAL).sum()
    error_rate = nErrors / float(LVAL.size) * 100

    return error_rate

def compute_error_rates_multi(PVALs, LVAL):
    error_rates = [compute_error_rate(p, LVAL) for p in PVALs]
    return error_rates
        