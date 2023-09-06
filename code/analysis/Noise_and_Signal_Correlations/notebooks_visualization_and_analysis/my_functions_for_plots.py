import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from scipy.stats import ks_2samp

#FUNCTIONS FOR PLOTS

#distributions of noise correlations, signal correlations and pearson coeffs
def plot_and_test_histograms(data1, data2, nbins = 20, num_permutations=1000, alpha=0.05, xlabel = ' ', title_plt = ' ', vertical_mode_lines = False, save_state = False, filename = 'p.png'):
    """
    Compare two datasets using histograms and a Kolmogorov-Smirnov test with permutations.
    
    Parameters:
    - data1: NumPy array or list, the first dataset.
    - data2: NumPy array or list, the second dataset.
    - num_permutations: Number of permutations for the KS test (default is 1000).
    - alpha: Significance level for the test (default is 0.05).

    Returns:
    - None (displays the histogram plots and test results).
    """
    
    # Create histograms for both datasets
    plt.hist(data1, bins=nbins, alpha=0.5, label='ACTIVE', density=True)
    plt.hist(data2, bins=nbins, alpha=0.5, label='PASSIVE', density=True)
    
    # Fit KDEs for both datasets and plot them
    kde_data1 = gaussian_kde(data1)
    kde_data2 = gaussian_kde(data2)
    
    x_values = np.linspace(min(min(data1), min(data2)), max(max(data1), max(data2)), 1000)
    plt.plot(x_values, kde_data1(x_values), 'r-', label='KDE ACTIVE', linewidth=3)
    plt.plot(x_values, kde_data2(x_values), 'g-', label='KDE PASSIVE', linewidth=3)
    
    if vertical_mode_lines == True:
        smooth_curve_data_1 = kde_data1(x_values)
        max_smooth_data_1 = np.argmax(smooth_curve_data_1)
        
        smooth_curve_data_2 = kde_data2(x_values)
        max_smooth_data_2 = np.argmax(smooth_curve_data_2)
        
        plt.axvline(x_values[max_smooth_data_1], color='red', linestyle='dashed', linewidth=2)
        plt.axvline(x_values[max_smooth_data_2], color='g', linestyle='dashed', linewidth=2)


    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.title(title_plt)
    
    
    # Perform KS test with permutations
    ks_statistic, p_value = ks_2samp(data1, data2)
    
    # Generate permuted datasets for the KS test
    combined_data = np.concatenate((data1, data2))
    permuted_ks_stats = []

    for _ in range(num_permutations):
        np.random.shuffle(combined_data)
        permuted_data1 = combined_data[:len(data1)]
        permuted_data2 = combined_data[len(data1):]
        permuted_ks_stat, _ = ks_2samp(permuted_data1, permuted_data2)
        permuted_ks_stats.append(permuted_ks_stat)
    
    # Calculate the p-value based on permutations
    permuted_ks_stats = np.array(permuted_ks_stats)
    p_value_permutation = (np.sum(permuted_ks_stats >= ks_statistic) + 1) / (num_permutations + 1)
    
    # Display KS test results
    print(f'KS Statistic: {ks_statistic:.4f}')
    print(f'KS Test p-value (permutations): {p_value_permutation:.4f}')
    print(f'Significance Level (alpha): {alpha:.4f}')

    # Determine if the datasets are statistically different
    if p_value_permutation < alpha:
        print('The two datasets are statistically different.')
    else:
        print('There is no significant difference between the two datasets.')
    
    if save_state == True:
        
        plt.savefig(filename)
    
    plt.show()
    
def permutation_test_histograms(data1, data2, num_permutations=10000, alpha=0.05, title = ' ', save_state = False, filename = 'p.png'):
    """
    Perform a permutation test to compare two histograms using the Kolmogorov-Smirnov statistic.

    Parameters:
    data1 (numpy.ndarray): First dataset (1D array).
    data2 (numpy.ndarray): Second dataset (1D array).
    num_permutations (int): Number of permutations to perform (default: 10000).
    alpha (float): Significance level (default: 0.05).

    Returns:
    float: The p-value for the permutation test.
    bool: True if the two histograms are statistically different, False otherwise.
    """
    # Compute the observed test statistic (Kolmogorov-Smirnov statistic)
    observed_statistic, _ = ks_2samp(data1, data2)

    # Initialize an array to store permuted test statistics
    permuted_statistics = np.zeros(num_permutations)

    # Combine the two datasets
    combined_data = np.concatenate([data1, data2])

    # Perform the permutation test
    for i in range(num_permutations):
        # Randomly permute the combined data
        np.random.shuffle(combined_data)

        # Split the permuted data back into two datasets
        permuted_data1 = combined_data[:len(data1)]
        permuted_data2 = combined_data[len(data1):]

        # Compute the test statistic for the permuted data
        permuted_statistic, _ = ks_2samp(permuted_data1, permuted_data2)

        # Store the permuted statistic
        permuted_statistics[i] = permuted_statistic

    # Calculate the p-value by comparing the observed statistic to the null distribution
    p_value = (np.sum(permuted_statistics >= observed_statistic) + 1) / (num_permutations + 1)

    # Plot the null distribution and observed statistic
    plt.hist(permuted_statistics, bins=30, color='gray', alpha=0.5, label='Null Distribution')
    plt.axvline(observed_statistic, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.xlabel('Test Statistic', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.title(title)
    
    if save_state == True:
        
        plt.savefig(filename)
    
    plt.show()

    # Print the p-value
    print(f'p-value: {p_value:.4f}')

    # Determine significance
    if p_value < alpha:
        return p_value, True
    else:
        return p_value, False
    
def plot_histogram(data, xlabel='', title='', num_bins=10, save_state = False, filename = 'p.png'):
    """
    Plot a histogram of the given data.

    Parameters:
    data (numpy.ndarray or list): The data to create the histogram from.
    xlabel (str): Label for the x-axis (optional).
    title (str): Title for the histogram (optional).
    num_bins (int): Number of bins for the histogram (default: 10).

    Returns:
    None
    """
    plt.hist(data, bins=num_bins, color='skyblue', alpha=0.7)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(title)
    #plt.grid(axis='y', alpha=0.7)
    
    if save_state == True:
        
        plt.savefig(filename)
    
    plt.show()
    
def fit_and_plot_histogram(data, xlabel='', title='', num_bins=10, bandwidth=0.5, save_state = False, filename = 'p.png'):
    """
    Fit and plot a smoothed histogram (KDE) of the given data.

    Parameters:
    data (numpy.ndarray or list): The data to create the histogram and KDE from.
    xlabel (str): Label for the x-axis (optional).
    title (str): Title for the histogram (optional).
    num_bins (int): Number of bins for the initial histogram (default: 10).
    bandwidth (float): Bandwidth parameter for kernel density estimation (KDE).

    Returns:
    None
    """
    # Create a histogram to visualize the data
    plt.hist(data, bins=num_bins, color='skyblue', alpha=0.7, density=True, label='Histogram')

    # Fit a kernel density estimate (KDE) to the data
    kde = gaussian_kde(data, bw_method=bandwidth)

    # Generate a smooth curve using the KDE
    x_values = np.linspace(min(data), max(data), 1000)
    smooth_curve = kde(x_values)
    
    max_smooth = np.argmax(smooth_curve)
    #plt.axvline(x_values[max_smooth], color='red', linestyle='dashed', linewidth=2)

    # Plot the smooth curve
    plt.plot(x_values, smooth_curve, 'r-', lw=3, label='KDE')

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Density',fontsize=14)
    plt.title(title)
    plt.legend()
    #plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    
    if save_state == True:
    
        plt.savefig(filename)
    
    plt.show()