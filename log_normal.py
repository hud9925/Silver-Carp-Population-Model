import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Function to plot the log-normal distribution based on sigma and target mean

def plot_lognormal_from_mean_sigma(target_mean, sigma, sample_size=10000):
    # Calculate mu from the target mean and sigma
    mu = np.log(target_mean) - (sigma**2 / 2)
    print(mu)
    # 12.37890799853647

    # Generate samples
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=sample_size)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    count, bins, ignored = ax.hist(samples, bins=50, density=True, alpha=0.6, color='g')
    
    # Try to fit line
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    ax.plot(x, p, 'k', linewidth=2)
    
    title = f"Log-normal Distribution\nMean = {target_mean}, σ = {sigma}"
    ax.set_title(title)
    ax.set_xlabel('Fecundity (number of eggs)')
    ax.set_ylabel('Density')
    plt.grid(True)
    plt.show()

# Initial guess values for demonstration
target_mean = 269388
# testing different sigma values
sigma_values = [0.5, 1, 1.5, 2] 
# 0.5 appears to be the best fit  
# Plot distributions for different sigma values
for sigma in sigma_values:
    plot_lognormal_from_mean_sigma(target_mean, sigma)