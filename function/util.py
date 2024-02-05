import torch 
import numpy as np
import os
import random
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def get_data_path():
    """
    Retrieve the path to the 'data' folder relative to the current file.

    Returns:
    - data_path (str): Path to the 'data' folder.
    """
    folder = os.path.dirname(__file__)
    return os.path.join(folder, "data")

def RSE(ypred, ytrue):
    """
    Calculate the Relative Squared Error (RSE) between predicted and true values.

    Args:
    - ypred (array-like): Predicted values.
    - ytrue (array-like): True values.

    Returns:
    - rse (float): Relative Squared Error.
    """
    rse = np.sqrt(np.square(ypred - ytrue).sum()) / \
            np.sqrt(np.square(ytrue - ytrue.mean()).sum())
    return rse

def quantile_loss(ytrue, ypred, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    '''
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()

def SMAPE(ytrue, ypred):
    """
    Compute the quantile loss version 2.

    Args:
    - ytrue (array-like): True values of shape (batch_size, output_horizon).
    - ypred (array-like): Predicted values of shape (batch_size, output_horizon, num_quantiles).
    - qs (list): List of quantiles.

    Returns:
    - L (float): Mean quantile loss.
    """
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel() + 1e-4
    mean_y = (ytrue + ypred) / 2.
    return np.mean(np.abs((ytrue - ypred) \
        / mean_y))

def MAPE(ytrue, ypred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Args:
    - ytrue (array-like): True values.
    - ypred (array-like): Predicted values.

    Returns:
    - mape (float): Mean Absolute Percentage Error.
    """
    ytrue = np.array(ytrue).ravel() + 1e-4
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) \
        / ytrue))

def train_test_split(X, y, train_ratio=0.6):
    """
    Split the input data into training and testing sets.

    Args:
    - X (array-like): Input data with shape (num_samples, num_periods, num_features).
    - y (array-like): Target data with shape (num_samples, num_periods).
    - train_ratio (float, optional): Ratio of data to be used for training. Default is 0.6.

    Returns:
    - Xtr (array-like): Training input data.
    - ytr (array-like): Training target data.
    - Xte (array-like): Testing input data.
    - yte (array-like): Testing target data.
    """
    num_ts, num_periods, num_features = X.shape
    train_periods = int(num_periods * train_ratio)
    random.seed(2)
    Xtr = X[:, :train_periods, :]
    ytr = y[:, :train_periods]
    Xte = X[:, train_periods:, :]
    yte = y[:, train_periods:]
    return Xtr, ytr, Xte, yte

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Attributes:
    - mean (float): The mean value of the data.
    - std (float): The standard deviation of the data.

    Methods:
    - fit_transform(y): Compute the mean and standard deviation of y and return standardized y.
    - inverse_transform(y): Transform standardized y back to the original scale.
    - transform(y): Standardize y based on the computed mean and standard deviation.
    """
    
    def fit_transform(self, y):
        """
        Compute the mean and standard deviation of y and return standardized y.

        Args:
        - y (array-like): Input data.

        Returns:
        - array-like: Standardized data.
        """
        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-4
        return (y - self.mean) / self.std
    
    def inverse_transform(self, y):
        """
        Transform standardized y back to the original scale.

        Args:
        - y (array-like): Standardized data.

        Returns:
        - array-like: Data in the original scale.
        """
        return y * self.std + self.mean

    def transform(self, y):
        """
        Standardize y based on the computed mean and standard deviation.

        Args:
        - y (array-like): Input data.

        Returns:
        - array-like: Standardized data.
        """
        return (y - self.mean) / self.std

class MaxScaler:
    """
    Scale features by dividing each feature by its maximum value.

    Attributes:
    - max (float): The maximum value of the data.

    Methods:
    - fit_transform(y): Compute the maximum value of y and return scaled y.
    - inverse_transform(y): Transform scaled y back to the original scale.
    - transform(y): Scale y based on the computed maximum value.
    """

    def fit_transform(self, y):
        """
        Compute the maximum value of y and return scaled y.

        Args:
        - y (array-like): Input data.

        Returns:
        - array-like: Scaled data.
        """
        self.max = np.max(y)
        return y / self.max
    
    def inverse_transform(self, y):
        """
        Transform scaled y back to the original scale.

        Args:
        - y (array-like): Scaled data.

        Returns:
        - array-like: Data in the original scale.
        """
        return y * self.max

    def transform(self, y):
        """
        Scale y based on the computed maximum value.

        Args:
        - y (array-like): Input data.

        Returns:
        - array-like: Scaled data.
        """
        return y / self.max


class MeanScaler:
    """
    Scale features by dividing each feature by its mean value.

    Attributes:
    - mean (float): The mean value of the data.

    Methods:
    - fit_transform(y): Compute the mean value of y and return scaled y.
    - inverse_transform(y): Transform scaled y back to the original scale.
    - transform(y): Scale y based on the computed mean value.
    """
    
    def fit_transform(self, y):
        """
        Compute the mean value of y and return scaled y.

        Args:
        - y (array-like): Input data.

        Returns:
        - array-like: Scaled data.
        """
        self.mean = np.mean(y)
        return y / self.mean
    
    def inverse_transform(self, y):
        """
        Transform scaled y back to the original scale.

        Args:
        - y (array-like): Scaled data.

        Returns:
        - array-like: Data in the original scale.
        """
        return y * self.mean

    def transform(self, y):
        """
        Scale y based on the computed mean value.

        Args:
        - y (array-like): Input data.

        Returns:
        - array-like: Scaled data.
        """
        return y / self.mean

class LogScaler:
    """
    Scale features using natural logarithm transformation.

    Methods:
    - fit_transform(y): Transform y using natural logarithm.
    - inverse_transform(y): Transform y back to the original scale.
    - transform(y): Transform y using natural logarithm.
    """

    def fit_transform(self, y):
        """
        Transform y using natural logarithm.

        Args:
        - y (array-like): Input data.

        Returns:
        - array-like: Transformed data.
        """
        return np.log1p(y)
    
    def inverse_transform(self, y):
        """
        Transform y back to the original scale.

        Args:
        - y (array-like): Transformed data.

        Returns:
        - array-like: Data in the original scale.
        """
        return np.expm1(y)

    def transform(self, y):
        """
        Transform y using natural logarithm.

        Args:
        - y (array-like): Input data.

        Returns:
        - array-like: Transformed data.
        """
        return np.log1p(y)


def gaussian_likelihood_loss(z, mu, sigma):
    """
    Calculate the Gaussian likelihood loss.

    Args:
    - z (tensor): True observations, shape (num_ts, num_periods).
    - mu (tensor): Mean, shape (num_ts, num_periods).
    - sigma (tensor): Standard deviation, shape (num_ts, num_periods).

    Returns:
    - tensor: Negative log likelihood loss.
    
    Likelihood formula:
    (2 * pi * sigma^2)^(-1/2) * exp(-(z - mu)^2 / (2 * sigma^2))

    Log likelihood formula:
    -1/2 * (log(2 * pi) + 2 * log(sigma)) - (z - mu)^2 / (2 * sigma^2)
    """
    negative_likelihood = torch.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6
    return negative_likelihood.mean()

def negative_binomial_loss(ytrue, mu, alpha):
    """
    Calculate the negative binomial loss.

    Args:
    - ytrue (array like): True observations.
    - mu (array like): Mean.
    - alpha (array like): Alpha parameter.

    Returns:
    - tensor: Negative log likelihood loss.
    
    Negative binomial likelihood formula:
    log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
    - 1 / alpha * log(1 + alpha * mu) + z * log(alpha * mu / (1 + alpha * mu))

    Minimize loss = -log l_{nb}

    Note: torch.lgamma calculates the log Gamma function.
    """
    batch_size, seq_len = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return - likelihood.mean()

def gamma_likelihood_loss(z, alpha, beta):
    """
    Calculate the Gamma likelihood loss.

    Args:
    - z (tensor): True observations, shape (num_ts, num_periods).
    - alpha (tensor): Shape parameter, shape (num_ts, num_periods).
    - beta (tensor): Scale parameter, shape (num_ts, num_periods).

    Returns:
    - tensor: Negative log likelihood loss.
    
    Gamma likelihood formula:
    z**(alpha-1) * exp(-beta*z) * beta**alpha / Gamma(alpha) 

    Log likelihood formula:
    (alpha-1)*log(z) - beta*z + alpha*log(beta) - log(Gamma(alpha))
    """
    log_unnormalized_prob = torch.xlogy(alpha - 1., z) - beta * z
    log_normalization = torch.lgamma(alpha) - alpha * torch.log(beta)
    likelihood = log_unnormalized_prob - log_normalization 
    return -likelihood.mean()

def Betaprm_likelihood_loss(z, alpha, beta):
    """
    Calculate the Beta prime likelihood loss.

    Args:
    - z (tensor): True observations, shape (num_ts, num_periods).
    - alpha (tensor): Shape parameter, shape (num_ts, num_periods).
    - beta (tensor): Scale parameter, shape (num_ts, num_periods).

    Returns:
    - tensor: Negative log likelihood loss.

    Beta prime likelihood formula:
    z**(alpha-1) * (1+z)**-(alpha+beta) / Beta(alpha,beta)

    Log likelihood formula:
    (alpha-1)*log(z) - (alpha+beta)*log(z+1) - log(Beta(alpha, beta))
    """
    likelihood = torch.lgamma(alpha+beta)-torch.lgamma(alpha)-torch.lgamma(beta)+ torch.xlogy(alpha - 1., z)- torch.xlogy(alpha + beta, z+1)
    return -likelihood.mean()

def Igamma_likelihood_loss1(z, alpha, beta):
    """
    Calculate the Incomplete Gamma likelihood loss version 1.

    Args:
    - z (tensor): True observations, shape (num_ts, num_periods).
    - alpha (tensor): Shape parameter, shape (num_ts, num_periods).
    - beta (tensor): Scale parameter, shape (num_ts, num_periods).

    Returns:
    - tensor: Negative log likelihood loss.

    Incomplete Gamma likelihood formula:
    z**(alpha-1) * (1+z)**-(alpha+beta) / Beta(alpha,beta)  

    Log likelihood formula:
    (alpha-1)*log(z) - (alpha+beta)*log(z+1) - log(Beta(alpha, beta))
    """
    likelihood = torch.xlogy(alpha , beta) - torch.lgamma(alpha) - torch.xlogy(1+alpha, z) - (beta/z)
    return -likelihood.mean()

def Igamma_likelihood_loss(z, alpha, beta):
    """
    Calculate the Incomplete Gamma likelihood loss.

    Args:
    - z (tensor): True observations, shape (num_ts, num_periods).
    - alpha (tensor): Shape parameter, shape (num_ts, num_periods).
    - beta (tensor): Scale parameter, shape (num_ts, num_periods).

    Returns:
    - tensor: Negative log likelihood loss.

    Incomplete Gamma likelihood formula:
    -alpha * log(z) - log(Gamma(alpha)) - (1/z)

    Note:
    - The term -alpha * log(z) represents the incomplete gamma distribution.
    - The term -log(Gamma(alpha)) is the logarithm of the gamma function.
    - The term -(1/z) accounts for the scaling factor.

    """

    likelihood = -torch.xlogy(alpha+1 , z) - torch.lgamma(alpha) - (1/z)
    return -likelihood.mean()

def Igaussian_likelihood_loss(z, mu, sigma):
    """
    Calculate the Inverse Gaussian likelihood loss.

    Args:
    - z (tensor): True observations, shape (num_ts, num_periods).
    - mu (tensor): Mean, shape (num_ts, num_periods).
    - sigma (tensor): Standard deviation, shape (num_ts, num_periods).

    Returns:
    - tensor: Negative log likelihood loss.

    Inverse Gaussian likelihood formula:
    - 0.5 * (3 * log(z + 1) + log(2*pi)) + (z - mu) ** 2 / (z * 2 * (mu ** 2))

    Note:
    - The term 0.5 * (3 * log(z + 1) + log(2*pi)) is the logarithm of the Inverse Gaussian distribution.
    - The term (z - mu) ** 2 / (z * 2 * (mu ** 2)) is the squared difference between true observations and mean.

    """

    negative_likelihood = torch.log(z + 1) + (z - mu) ** 2 / (z* 2 * (mu ** 2)) + 6 
   # negative_likelihood = 0.5*( 3* torch.log(z + 1) + math.log(2*math.pi) ) + (z - mu) ** 2 / (z* 2 * (mu ** 2)) 
    return negative_likelihood.mean()

def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    """
    Generate batches of training data for sequence models.

    Args:
    - X (array like): Input data, shape (num_samples, num_features, num_periods).
    - y (array like): Target data, shape (num_samples, num_periods).
    - num_obs_to_train (int): Number of observations to use for training.
    - seq_len (int): Length of the sequence/encoder/decoder.
    - batch_size (int): Size of the batch.

    Returns:
    - tuple: Tuple containing X_train_batch, y_train_batch, Xf, yf.

    Explanation:
    - X_train_batch: Input training batch data, shape (batch_size, num_obs_to_train, num_features).
    - y_train_batch: Target training batch data, shape (batch_size, num_obs_to_train).
    - Xf: Input forward batch data, shape (batch_size, seq_len, num_features).
    - yf: Target forward batch data, shape (batch_size, seq_len).

    """
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t-num_obs_to_train:t, :]
    y_train_batch = y[batch, t-num_obs_to_train:t]
    Xf = X[batch, t:t+seq_len]
    yf = y[batch, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf


def compute_quantile_loss(y_true, y_pred, quantile):
    """
    Compute the quantile loss between true and predicted values.

    Parameters
    ----------
    y_true : 1d ndarray
        Target values.

    y_pred : 1d ndarray
        Predicted values.

    quantile : float, 0. ~ 1.
        Quantile to be evaluated, e.g., 0.5 for median.

    Returns
    -------
    loss : 1d ndarray
        Quantile loss computed for each data point.
        
    Explanation
    -----------
    The quantile loss is calculated as the maximum of two terms:
    1. quantile * residual if residual > 0
    2. (quantile - 1) * residual if residual <= 0
    where residual = y_true - y_pred.

    """
    residual = y_true - y_pred
    return np.maximum(quantile * residual, (quantile - 1) * residual)



def plot_train_test_sample(dataframes, train_start, train_end, test_start, test_end):
    """
    Plot the training and test data samples for each variable in the given DataFrames.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        List of DataFrames containing the data to be plotted.

    train_start : str
        Start date of the training data.

    train_end : str
        End date of the training data.

    test_start : str
        Start date of the test data.

    test_end : str
        End date of the test data.

    Returns
    -------
    None
        This function plots the training and test data samples for each variable and saves the plot as a JPEG image.
    """
    # Convert 'date' column to datetime type and set 'date' column as index for each DataFrame
    for df in dataframes:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Titles for each subplot
    titles = ['FPI', 'Cereals', 'Dairy', 'Oils']

    # Plot the training and test curves for each variable on their respective subplots
    for idx, df in enumerate(dataframes):
        tr_data = df.loc[train_start:train_end]
        ts_data = df.loc[test_start:test_end]

        row = idx // 2
        col = idx % 2

        axes[row, col].plot(tr_data.index, tr_data['load'], label="Train Data", color='blue')
        axes[row, col].plot(ts_data.index, ts_data['load'], label="Test Data", color='red', ls='--')

        # Add legends to each subplot
        axes[row, col].legend()

        # Add grid to each subplot
        axes[row, col].grid(True)

        # Set titles for each subplot
        font = {'weight': 'bold', 'size': 16}
        axes[row, col].set_title(titles[idx], fontdict=font)

    # Adjust layout to avoid overlapping
    plt.tight_layout()

    # Save the plot as a JPEG image
    plt.savefig('train_test.jpeg', format='jpeg', dpi=300)

    # Display the plots
    plt.show()
    
