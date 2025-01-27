import numpy

def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)     # precision matrix
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

def compute_ll(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

