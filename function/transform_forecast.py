

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
import tqdm as tq
from tqdm import tqdm
import pandas as pd
import util
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from time import time
import argparse
from datetime import date
from progressbar import *

import holidays



"""# Transfprmer"""

class Backbone(nn.Module):
    """
    Backbone module for the neural network architecture.

    This module processes the input data through a transformer layer and a linear layer.

    Parameters:
    - global_hidden_size (int): The hidden size of the transformer layer.
    - n_global_factors (int): The number of global factors.
    - num_layers (int): The number of layers in the transformer.

    Attributes:
    - transformer (TransAm): The transformer layer.
    - factor (nn.Linear): The linear layer.

    Methods:
    - forward(X): Forward pass of the Backbone module.

    """

    def __init__(self, global_hidden_size, n_global_factors, num_layers):
        """
        Initializes the Backbone module with the specified parameters.

        Parameters:
        - global_hidden_size (int): The hidden size of the transformer layer.
        - n_global_factors (int): The number of global factors.
        - num_layers (int): The number of layers in the transformer.
        """
        super(Backbone, self).__init__()
        self.transformer = TransAm(global_hidden_size)
        self.factor = nn.Linear(global_hidden_size, n_global_factors)

    def forward(self, X):
        """
        Performs a forward pass of the Backbone module.

        Parameters:
        - X (tensor): The input tensor of shape (num_ts, num_features).

        Returns:
        - gt (tensor): The output tensor of shape (num_ts, n_global_factors).
        """
        num_ts, num_features = X.shape
        X = X.unsqueeze(1)
        ht = self.transformer(X)
        ht = F.relu(ht)
        gt = ht
        return gt.view(num_ts, -1)

class Noise(nn.Module):
    """
    Noise module for the neural network architecture.

    This module processes the input data through a transformer layer and a linear layer to predict noise.

    Parameters:
    - noise_hidden_size (int): The hidden size of the transformer layer.
    - num_layers (int): The number of layers in the transformer.

    Attributes:
    - transformer (TransAm): The transformer layer.
    - affine (nn.Linear): The linear layer for predicting noise.

    Methods:
    - forward(X): Forward pass of the Noise module.

    """

    def __init__(self, noise_hidden_size, num_layers):
        """
        Initializes the Noise module with the specified parameters.

        Parameters:
        - noise_hidden_size (int): The hidden size of the transformer layer.
        - num_layers (int): The number of layers in the transformer.
        """
        super(Noise, self).__init__()
        self.transformer = TransAm(noise_hidden_size, num_layers)
        self.affine = nn.Linear(noise_hidden_size, 1)

    def forward(self, X):
        """
        Performs a forward pass of the Noise module.

        Parameters:
        - X (tensor): The input tensor of shape (num_ts, num_features).

        Returns:
        - sigma_t (tensor): The predicted noise tensor of shape (num_ts, 1).
        """
        num_ts, num_features = X.shape
        X = X.unsqueeze(1)
        ht = self.transformer(X)
        ht = F.relu(ht)
        sigma_t = self.affine(ht)
        sigma_t = torch.log(1 + torch.exp(sigma_t))
        return sigma_t.view(-1, 1)

class TDPnet(nn.Module):
    """
    Temporal Dependency Network (TDPnet) module for time series forecasting.

    This module predicts the mean and variance of the time series using a global factor backbone and a noise module.

    Parameters:
    - noise_hidden_size (int): The hidden size of the noise module.
    - global_hidden_size (int): The hidden size of the global factor backbone.
    - n_global_factors (int): The number of global factors.
    - num_layers (int): The number of layers in the modules.

    Attributes:
    - noise (Noise): The noise module.
    - global_factor (Backbone): The global factor backbone.
    - embed (nn.Linear): The linear layer for embedding global factors.

    Methods:
    - forward(X): Forward pass of the TDPnet module.
    - sample(X, num_samples): Generate samples from the TDPnet module.
    """

    def __init__(self, noise_hidden_size, global_hidden_size, n_global_factors, num_layers):
        """
        Initializes the TDPnet module with the specified parameters.

        Parameters:
        - noise_hidden_size (int): The hidden size of the noise module.
        - global_hidden_size (int): The hidden size of the global factor backbone.
        - n_global_factors (int): The number of global factors.
        - num_layers (int): The number of layers in the modules.
        """
        super(TDPnet, self).__init__()
        self.noise = Noise(noise_hidden_size, num_layers)
        self.global_factor = Backbone(global_hidden_size, n_global_factors, num_layers)
        self.embed = nn.Linear(global_hidden_size, n_global_factors)

    def forward(self, X,):
        """
        Performs a forward pass of the TDPnet module.

        Parameters:
        - X (tensor): The input tensor of shape (num_ts, num_periods, num_features).

        Returns:
        - mu (tensor): The predicted mean of the time series, tensor of shape (num_ts, num_periods).
        - sigma (tensor): The predicted variance of the time series, tensor of shape (num_ts, num_periods).
        """
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
        num_ts, num_periods, num_features = X.size()
        mu = []
        sigma = []
        for t in range(num_periods):
            gt = self.global_factor(X[:, t, :])
            ft = self.embed(gt)
            ft = ft.sum(dim=1).view(-1, 1)
            sigma_t = self.noise(X[:, t, :])
            mu.append(ft)
            sigma.append(sigma_t)
        mu = torch.cat(mu, dim=1).view(num_ts, num_periods)
        sigma = torch.cat(sigma, dim=1).view(num_ts, num_periods) + 1e-6
        return mu, sigma

    def sample(self, X, num_samples=100):
        """
        Generate samples from the TDPnet module.

        Parameters:
        - X (tensor): The input tensor of shape (num_ts, num_periods, num_features).
        - num_samples (int): The number of samples to generate.

        Returns:
        - z (tensor): The generated samples, tensor of shape (num_ts, num_periods).
        """
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
        mu, var = self.forward(X)
        num_ts, num_periods = mu.size()
        z = torch.zeros(num_ts, num_periods)
        for _ in range(num_samples):
            dist = torch.distributions.normal.Normal(loc=mu, scale=var)
            zs = dist.sample().view(num_ts, num_periods)
            z += zs
        z = z / num_samples
        return z

def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    '''
    Generate batches of training data for a sequence-to-sequence model.

    Args:
    - X (array-like): Input data with shape (num_samples, num_features, num_periods).
    - y (array-like): Target data with shape (num_samples, num_periods).
    - num_obs_to_train (int): Number of observations used for training.
    - seq_len (int): Length of the sequence/encoder/decoder.
    - batch_size (int): Size of the batch.

    Returns:
    - X_train_batch (array-like): Training input batch with shape (batch_size, num_obs_to_train, num_features).
    - y_train_batch (array-like): Training target batch with shape (batch_size, num_obs_to_train).
    - Xf (array-like): Input sequence for forecasting with shape (batch_size, seq_len, num_features).
    - yf (array-like): Target sequence for forecasting with shape (batch_size, seq_len).
    '''
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

def train(
    X,
    y,
    args
    ):
    '''
    Train a time series forecasting model using the provided data.

    Args:
    - X (array-like): Input data with shape (num_samples, num_features, num_periods).
    - y (array-like): Target data with shape (num_samples, num_periods).
    - args (dict): A dictionary containing various training parameters:
        - num_epoches (int): Number of epochs to run.
        - step_per_epoch (int): Steps per epoch to run.
        - seq_len (int): Output horizon.
        - likelihood (str): Type of likelihood to use; default is Gaussian.
        - num_skus_to_show (int): Number of SKUs to show in the test phase.
        - num_results_to_sample (int): Number of samples in the test phase for prediction.
        - noise_hidden_size (int): Size of the hidden layer for noise.
        - global_hidden_size (int): Size of the hidden layer for global factors.
        - n_factors (int): Number of global factors.
        - Transformerheads (int): Number of transformer heads.
        - lr (float): Learning rate.
        - standard_scaler (bool): Whether to use standard scaler for scaling y.
        - log_scaler (bool): Whether to use log scaler for scaling y.
        - mean_scaler (bool): Whether to use mean scaler for scaling y.
        - batch_size (int): Size of the batch.
        - num_obs_to_train (int): Number of observations used for training.
        - sample_size (int): Size of the sample for prediction.
        - show_plot (bool): Whether to show the plot.

    Returns:
    - losses (list): List of losses during training.
    - mape_list (list): List of Mean Absolute Percentage Error (MAPE) values.
    - yscaler: Scaler object used for y scaling.
    - model: Trained TDPnet model.
    - Xte: Test input data.
    - yte: Test target data.
    '''
    # rho = args.quantile
    num_ts, num_periods, num_features = X.shape
    model = TDPnet(
        args["noise_hidden_size"],args["global_hidden_size"], args["n_factors"], args["Transformerheads"])
    optimizer = Adam(model.parameters(), lr=args["lr"])

    random.seed(2)
    # select sku with most top n quantities
    Xtr, ytr, Xte, yte = util.train_test_split(X, y)
    losses = []
    cnt = 0

    yscaler = None
    if args["standard_scaler"]:
        yscaler = util.StandardScaler()
    elif args["log_scaler"]:
        yscaler = util.LogScaler()
    elif args["mean_scaler"]:
        yscaler = util.MeanScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    # training
    progress = ProgressBar()
    seq_len = args["seq_len"]
    num_obs_to_train = args["num_obs_to_train"]
    for epoch in progress(range(args["num_epoches"])):
        for step in range(args["step_per_epoch"]):
            Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train,
                        seq_len, args["batch_size"])
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()
            yf = torch.from_numpy(yf).float()
            mu, sigma = model(Xtrain_tensor)
            loss = util.gaussian_likelihood_loss(ytrain_tensor, mu, sigma)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1


    # test
    mape_list = []
    # select skus with most top K
    X_test = Xte[:, -seq_len-num_obs_to_train:-seq_len, :].reshape((num_ts, -1, num_features))
    Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    y_test = yte[:, -seq_len-num_obs_to_train:-seq_len].reshape((num_ts, -1))
    yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
    if yscaler is not None:
        y_test = yscaler.transform(y_test)

    result = []
    n_samples = args["sample_size"]
    for _ in tq.tqdm(range(n_samples)):
        y_pred = model.sample(Xf_test)
        y_pred = y_pred.data.numpy()
        if yscaler is not None:
            y_pred = yscaler.inverse_transform(y_pred)
        result.append(y_pred.reshape((-1, 1)))


    result = np.concatenate(result, axis=1)
    p50 = np.quantile(result, 0.5, axis=1)
    p90 = np.quantile(result, 0.9, axis=1)
    p10 = np.quantile(result, 0.1, axis=1)
    print("p50 {}".format(p50))

    mape = util.MAPE(yf_test, p50)
    print("P50 MAPE: {}".format(mape))
    mape_list.append(mape)

    if args["show_plot"]:
        plt.figure(1, figsize=(20, 5))
        plt.plot([k + seq_len + num_obs_to_train - seq_len \
            for k in range(seq_len)], p50, "r-")
        plt.fill_between(x=[k + seq_len + num_obs_to_train - seq_len for k in range(seq_len)], \
            y1=p10, y2=p90, alpha=0.5)
        plt.title('Prediction uncertainty')
        yplot = yte[-1, -seq_len-num_obs_to_train:]
        plt.plot(range(len(yplot)), yplot, "k-")
        plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper left")
        ymin, ymax = plt.ylim()
        plt.vlines(seq_len + num_obs_to_train - seq_len, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.show()
    return losses, mape_list, yscaler, model, Xte, yte



def process_data(seq, parameters, datasets):
    """
    Process data for multiple datasets based on provided sequences and parameters.

    Args:
    - seq (list): List of sequences indicating the number of time periods to process.
    - parameters (list): List of dictionaries containing parameters for data processing and training.
    - datasets (list): List of datasets to process.

    Returns:
    dict: A dictionary containing processed data including p50, p90, p10, true values, and x_test.

    This function processes data for multiple datasets based on the sequences provided. It iterates over the datasets, processes each dataset using the parameters specified, and trains the data. After training, it predicts the values for each sequence in the test data using the trained model, computes the 50th, 90th, and 10th percentile values (p50, p90, p10), and stores them along with the true values and x_test data for each sequence.

    """
    p50_lists = {}
    p90_lists = {}
    p10_lists = {}
    true_values_lists = {}
    x_test_lists = {}

    for dataset, para, seq_len in zip(datasets, parameters, seq):
        p50_list = []
        p90_list = []
        p10_list = []
        true_values = []
        x_test_list = []

        processing = DataProcessor(para)
        X, y = processing.preprocess_data(dataset)
        losses, mape_list, yscaler, model, Xte, yte = train(X, y, para)
        num_ts, _, num_features = Xte.shape
        pp = (Xte.shape[1]) // seq_len

        for k in tq.tqdm(range(pp)):
            mape_list = []
            Xf_test = Xte[:, seq_len*k:seq_len*(k+1), :].reshape((num_ts, -1, num_features))
            yf_test = yte[:, seq_len*k:seq_len*(k+1)].reshape((num_ts, -1))

            result = []
            n_samples = para["sample_size"]
            for _ in range(n_samples):
                y_pred = model.sample(Xf_test)
                y_pred = y_pred.data.numpy()
                if yscaler is not None:
                    y_pred = yscaler.inverse_transform(y_pred)
                result.append(y_pred.reshape((-1, 1)))

            result = np.concatenate(result, axis=1)
            p50 = np.quantile(result, 0.5, axis=1)
            p90 = np.quantile(result, 0.9, axis=1)
            p10 = np.quantile(result, 0.1, axis=1)

            p50_list.append(p50)
            p90_list.append(p90)
            p10_list.append(p10)
            true_values.append(yf_test[0])
            x_test_list.append(Xf_test)

        p50_lists[f"p50_list_{dataset}"] = np.array(p50_list).flatten()
        p90_lists[f"p90_list_{dataset}"] = np.array(p90_list).flatten()
        p10_lists[f"p10_list_{dataset}"] = np.array(p10_list).flatten()
        true_values_lists[f"true_values_{dataset}"] = np.array(true_values).flatten()
        x_test_lists[f"x_test_list_{dataset}"] = x_test_list

    return {
        **p50_lists,
        **p90_lists,
        **p10_lists,
        **true_values_lists,
        **x_test_lists
    }




feature_size = 19
class TransAm(nn.Module):
    def __init__(self,global_hidden_size, num_layers=1,feature_size=feature_size,dropout=0.2):
        """
        Transformer-based model for sequence processing.

        Args:
        - global_hidden_size (int): Size of the hidden layer in the global context.
        - num_layers (int, optional): Number of encoder layers. Defaults to 1.
        - feature_size (int, optional): Size of the features. Defaults to feature_size.
        - dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        # self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=feature_size, dropout=dropout)
        ### how many encoder layers you want
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        ### decoder
        self.decoder = nn.Linear(feature_size, global_hidden_size)
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the decoder linear layer.
        """
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        """
        Forward pass of the Transformer-based model.

        Args:
        - src (torch.Tensor): Source input tensor.

        Returns:
        - output (torch.Tensor): Output tensor.
        """
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a square subsequent mask.

        Args:
        - sz (int): Size of the mask.

        Returns:
        - mask (torch.Tensor): Subsequent mask tensor.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask





class DataProcessor:
    def __init__(self, args):
        """
        Initialize the DataProcessor.

        Args:
        - args (dict): Dictionary containing arguments for data processing.
          Expected keys: 'num_obs_to_train', 'seq_len', 'lags'.
        """
        self.args = args
        self.num_obs_to_train = args["num_obs_to_train"]
        self.seq_len = args["seq_len"]

    def preprocess_data(self, filename):
        """
        Preprocesses the data from the given file.

        Reads the data from a CSV file, scales the 'load' column using MinMaxScaler,
        extracts additional features such as year, day of week, hour, and identifies holidays.
        Generates lag features based on specified lag values.
        Drops missing values and prepares the data for training.

        Args:
        - filename (str): Path to the CSV file containing the data.

        Returns:
        - X (numpy.ndarray): Input features array of shape (num_samples, num_periods, num_features).
        - y (numpy.ndarray): Target variable array of shape (num_samples, num_periods).
        """
        data = pd.read_csv(filename, parse_dates=["date"])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array(data.load).reshape(-1, 1))
        scaled_data = scaler.transform(np.array(data.load).reshape(-1, 1))
        data['load'] = np.array(scaled_data).reshape(1, -1)[0]
        data["year"] = data["date"].apply(lambda x: x.year)
        data["day_of_week"] = data["date"].apply(lambda x: x.dayofweek)
        data["hour"] = data["date"].apply(lambda x: x.day)
        data.set_index(pd.to_datetime(data['date']), inplace=True)
        
        import holidays 
        country_code = 'US'
        country_holidays = holidays.CountryHoliday(country_code)
        year = 1990
        month = 1
        monthly_holidays = [
            date
            for date in country_holidays
            if date.year == year and date.month == month
        ]

        cet_dates = pd.Series(data.index, index=data.index)
        data["holiday"] = cet_dates.apply(lambda d: d in monthly_holidays)
        data["holiday"] = data["holiday"].astype(int)

        counter = np.arange(1, 25)
        lags = self.args.get('lags', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21, 24])
        lag_cols = [f"conso_lag_{cn}" for cn in lags]
        for lag, lag_col in zip(lags, lag_cols):
            data[lag_col] = data["load"].shift(lag)

        data.dropna(inplace=True)

        features = ["hour", "day_of_week", "holiday"]
        hours = data["hour"]
        dows = data["day_of_week"]
        holi = data["holiday"]
        lags = data[lag_cols]
        X = np.c_[np.asarray(hours), np.asarray(dows), np.asarray(holi), np.asarray(lags)]
        num_features = X.shape[1]
        num_periods = len(data)
        X = np.asarray(X).reshape((-1, num_periods, num_features))
        y = np.asarray(data["load"]).reshape((-1, num_periods))

        return X, y
