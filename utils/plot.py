import os
import matplotlib.pyplot as plt
import numpy

def plot_hist(D0, D1, title, xlabel, ylabel, class0="Fake", class1="Genuine", nBins=50, save_disk = False, output_dir="./assets/outputs", output_name=None):
    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.hist(D0[:], bins = nBins, density = True, alpha = 0.4, label = class0, color="blue")
    plt.hist(D1[:], bins = nBins, density = True, alpha = 0.4, label = class1, color="orange")

    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if save_disk:
        os.makedirs(output_dir, exist_ok=True)
        # plt.savefig(f"{output_dir}/{output_name if output_name else "hist"}.pdf")
        plt.savefig(f"{output_dir}/{output_name if output_name else "hist"}.jpg")
    # plt.show()
    plt.close()
    
    
def plot_scatter_2_classes(D0, D1, title, xlabel, ylabel, class0, class1, save_disk=False, output_dir="./assets/outputs", output_name=None):
    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(D0[0], D0[1], label=class0, s=1 ).set(facecolor = "blue")
    plt.scatter(D1[0], D1[1], label=class1, s=1 ).set(facecolor = "orange")

    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if save_disk:
        os.makedirs(output_dir, exist_ok=True)
        # plt.savefig(f"{output_dir}/{output_name if output_name else "scatter_2_classes"}.pdf")
        plt.savefig(f"{output_dir}/{output_name if output_name else "scatter_2_classes"}.jpg")
    
    # plt.show()
    plt.close()

def plot_scatter_PCA(D0, D1, m, class0="Fake", class1="Genuine", save_disk=False, output_dir="./assets/outputs"):
  for dIdx1 in range(m):
    for dIdx2 in range(m):
        if dIdx2 <= dIdx1:
            continue
        xlabel = f"Direction {dIdx1}"
        ylabel = f"Direction {dIdx2}"
        output_name = f"PCA_scatter_{dIdx1}_{dIdx2}"
        F0 = [D0[dIdx1, :], D0[dIdx2, :]]
        F1 = [D1[dIdx1, :], D1[dIdx2, :]]
        plot_scatter_2_classes(F0, F1, "PCA directions comparison", xlabel, ylabel, class0, class1, save_disk, output_dir, output_name)


def plot_Gaussian_density(D, X, pdf, title, xlabel, ylabel, classLabel, colorHist, colorPdf, nBins=50, save_disk = False, output_dir="./assets/outputs", output_name=None):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.plot(X, pdf, linewidth=3, color=colorPdf)           # Gaussian pdf
    plt.hist(D[:], bins = nBins, density = True, alpha = 0.5, label = classLabel, color=colorHist)

    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if save_disk:
        os.makedirs(output_dir, exist_ok=True)
        # plt.savefig(f"{output_dir}/{output_name if output_name else "hist"}.pdf")
        plt.savefig(f"{output_dir}/{output_name if output_name else "hist_Gaussian"}.jpg")
    # plt.show()
    plt.close()


def plot_log(X, Y0, Y1=None, title="Log-scale plot", xlabel="X", ylabel="Y", label0="False", label1="True", save_disk=False, output_dir="./assets/outputs", output_name=None):
    if Y1 and len(Y0) != len(Y1):
        raise ValueError("Non consistent sizes in Y0 and Y1")
    
    # colors = plt.cm.cool(numpy.linspace(0, 1, len(Y0)))
    # colors = plt.cm.viridis(numpy.linspace(0, 1, len(Y0)))
    # colors = plt.cm.plasma(numpy.linspace(0, 1, len(Y0)))
    # colors = plt.cm.cividis(numpy.linspace(0, 1, len(Y0)))
    # colors = plt.cm.autumn(numpy.linspace(0, 1, len(Y0)))

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.figure(figsize=(10, 6))
    plt.xscale('log', base=10)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if isinstance(Y0, list) and all(not isinstance(x, list) for x in Y0):           # Y contain single lists
        plt.plot(X, Y0, marker='o', label=label0, color='blue', linewidth=1.6)
        if Y1:
            plt.plot(X, Y1, marker='o', label=label1, color='orange', linewidth=1.6)
        plt.legend(fontsize=12)
    else:                                                                           # Y contain many lists to be compared
        colors = plt.cm.plasma(numpy.linspace(0, 1, len(Y0)))
        for i in range(len(Y0)):
            plt.plot(X, Y0[i], marker='o', label=label0[i], color=colors[i], linewidth=1.6)
            if Y1:
                plt.plot(X, Y1[i], marker='o', label=label1[i], color=colors[i], linestyle='--', linewidth=1.3)
        plt.legend(fontsize=8)

    plt.tight_layout()
    
    if save_disk:
        os.makedirs(output_dir, exist_ok=True)
        # plt.savefig(f"{output_dir}/{output_name}.pdf")
        plt.savefig(f"{output_dir}/{output_name}.jpg")


def plot_density(X, pdf, title, xlabel, ylabel, save_disk = False, output_dir="./assets/outputs", output_name=None):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    

    plt.plot(X, pdf, linewidth=2, color="blue")           # Gaussian pdf
    #plt.hist(D[:], bins = nBins, density = True, alpha = 0.5, label = classLabel, color="red")

    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if save_disk:
        os.makedirs(output_dir, exist_ok=True)
        # plt.savefig(f"{output_dir}/{output_name if output_name else "hist"}.pdf")
        plt.savefig(f"{output_dir}/{output_name if output_name else "hist_Gaussian"}.jpg")
    plt.show()
    plt.close()