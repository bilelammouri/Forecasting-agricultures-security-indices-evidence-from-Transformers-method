
import numpy as np
import os
import random
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
 
    
def check_stationarity(timeseries):
    """
    Check the stationarity of a given time series using the Dickey-Fuller test.

    Parameters:
    - timeseries (pandas Series or array-like): The time series data to be tested for stationarity.

    Returns:
    - None

    This function performs the Dickey-Fuller test on the provided time series data.
    It prints the results of the test including the Test Statistic, p-value,
    number of lags used, number of observations used, and critical values.
    """
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    
    
    
def sarima_grid_search(y,seasonal_period):
    """
    Perform grid search to find optimal SARIMA parameters based on AIC.

    Parameters:
    - y (array-like): The time series data to be modeled.
    - seasonal_period (int): The seasonal period of the time series.

    Returns:
    - None

    This function performs a grid search to find the optimal SARIMA parameters
    based on the Akaike Information Criterion (AIC). It iterates over a range of
    p, d, q, and seasonal p, d, q parameters to find the combination that minimizes
    the AIC. The SARIMA model is fitted for each parameter combination, and the
    parameters with the minimum AIC are printed.
    """
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]

    mini = float('+inf')


    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal

#                 print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))    
    
    
def fit_sarimax_models(data_list, orders_list, seasonal_orders_list):
    """
    Fit SARIMAX models to the provided time series data.

    Parameters:
    - data_list (list of array-like): List containing time series data to be modeled.
    - orders_list (list of tuples): List containing SARIMAX order parameters for each time series.
    - seasonal_orders_list (list of tuples): List containing seasonal SARIMAX order parameters for each time series.

    Returns:
    - results (dict): A dictionary containing fitted SARIMAX models for each time series.
        Keys are variable names ('FPI', 'cereals', 'dairy', 'oils'), and values are fitted SARIMAX models.

    This function fits SARIMAX models to the provided time series data using the specified order parameters.
    It iterates over the data list along with corresponding SARIMAX order and seasonal order parameters,
    fits SARIMAX models for each time series, and stores the fitted models in a dictionary.
    """
    results = {}

    # Loop through the data and corresponding orders
    for data, order, seasonal_order, var_name in zip(data_list, orders_list, seasonal_orders_list, ['FPI', 'cereals', 'dairy', 'oils']):
        # Create SARIMAX model
        mod = sm.tsa.statespace.SARIMAX(data,
                                            order=order,
                                            seasonal_order=seasonal_order,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
        # Fit the model
        results[var_name] = mod.fit()

    return results



def get_predictions_and_confidence_intervals(results):
    """
    Get predictions and confidence intervals from fitted SARIMAX models.

    Parameters:
    - results (dict): A dictionary containing fitted SARIMAX models.
        Keys are variable names ('FPI', 'cereals', 'dairy', 'oils'), and values are fitted SARIMAX models.

    Returns:
    - predictions (dict): A dictionary containing predictions and confidence intervals for each time series.
        Keys are constructed based on the variable names ('pred_FPI', 'pred_cereals', 'pred_dairy', 'pred_oils')
        for predictions and ('pred_ci_FPI', 'pred_ci_cereals', 'pred_ci_dairy', 'pred_ci_oils') for confidence intervals.
        Values are prediction objects and confidence interval DataFrames.

    This function computes predictions and confidence intervals from fitted SARIMAX models.
    It iterates over the fitted models, generates predictions starting from the 266th time step, and computes
    confidence intervals for each prediction. The predictions and confidence intervals are stored in a dictionary
    for each time series variable.
    """
    predictions = {}

    # Loop through the results and compute predictions and confidence intervals
    for var_name, result in results.items():
        # Get predictions
        pred = result.get_prediction(start=266, dynamic=False)
        # Get confidence intervals
        pred_ci = pred.conf_int()
        # Store predictions and confidence intervals in a dictionary
        predictions[f"pred_{var_name}"] = pred
        predictions[f"pred_ci_{var_name}"] = pred_ci

    return predictions



# Call this function after pick the right(p,d,q) for SARIMA based on AIC
def sarima_eva(y,order,seasonal_order,seasonal_period,pred_date,y_to_test):
    """
    Evaluate SARIMA model's performance and generate forecasts.

    Parameters:
    - y (array-like): Time series data to be modeled.
    - order (tuple): SARIMA order parameters (p, d, q).
    - seasonal_order (tuple): Seasonal SARIMA order parameters (P, D, Q, s).
    - seasonal_period (int): Seasonal period of the time series.
    - pred_date (int): The start date for forecasting.
    - y_to_test (array-like): Test data for evaluating the model.

    Returns:
    - results (SARIMAResults): Fitted SARIMA model results.
    - y_forecasted (array-like): Forecasted values using SARIMA model with dynamic=False.
    - y_forecasted_dynamic (array-like): Forecasted values using SARIMA model with dynamic=True.

    This function fits a SARIMA model to the provided time series data and evaluates its performance.
    It prints a summary of the fitted model, plots diagnostics, computes one-step-ahead forecasts, calculates
    Root Mean Squared Error (RMSE) for both dynamic and non-dynamic forecasts, and visualizes the forecasts along
    with the observed data.

    The function returns the fitted SARIMA model results, one-step-ahead forecasts, and dynamic forecasts.
    """
    # fit the model
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    print(results.summary().tables[1])

    results.plot_diagnostics(figsize=(16, 8))
    plt.show()

    # The dynamic=False argument ensures that we produce one-step ahead forecasts,
    # meaning that forecasts at each point are generated using the full history up to that point. pd.to_datetime(pred_date)
    pred = results.get_prediction(start=pred_date, dynamic=False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    mse = ((y_forecasted - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = False {}'.format(seasonal_period,round(np.sqrt(mse), 2)))

    ax = y.plot(label='observed')
    y_forecasted.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    plt.legend()
    plt.show()

    # A better representation of our true predictive power can be obtained using dynamic forecasts.
    # In this case, we only use information from the time series up to a certain point,
    # and after that, forecasts are generated using values from previous forecasted time points.
    pred_dynamic = results.get_prediction(start=pred_date, dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    mse_dynamic = ((y_forecasted_dynamic - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = True {}'.format(seasonal_period,round(np.sqrt(mse_dynamic), 2)))

    ax = y.plot(label='observed')
    y_forecasted_dynamic.plot(label='Dynamic Forecast', ax=ax,figsize=(14, 7))
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')

    plt.legend()
    plt.show()

    return (results, y_forecasted, y_forecasted_dynamic)






# Call this function after pick the right(p,d,q) for SARIMA based on AIC
def sarima_article(y,order,seasonal_order,seasonal_period,pred_date,y_to_train, y_to_test):
    """
    Fit SARIMA models with different forecast steps and save forecasts to an Excel file.

    Parameters:
    - y (array-like): Time series data to be modeled.
    - order (tuple): SARIMA order parameters (p, d, q).
    - seasonal_order (tuple): Seasonal SARIMA order parameters (P, D, Q, s).
    - seasonal_period (int): Seasonal period of the time series.
    - pred_date (int): The start date for forecasting.
    - y_to_train (array-like): Training data for SARIMA models.
    - y_to_test (array-like): Test data for evaluating the model.

    Returns:
    - None

    This function fits SARIMA models with different forecast steps to the provided time series data.
    It saves the forecasts to an Excel file named 'output_arima.xlsx'.
    """
    # fit the model
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()

    pred = results.get_prediction(start=pred_date, dynamic=False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean

    forecast_steps = 1
    pred1 = []
    for i in range((int(len(y_to_test)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred1.extend(sarima_forecast)

    y_forecasted1 = pred1

    forecast_steps = 2
    pred2 = []
    for i in range((int((len(y_to_test)+1)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred2.extend(sarima_forecast)

    y_forecasted2 = pred2[:-1]

    forecast_steps = 3
    pred3 = []
    for i in range((int(len(y_to_test)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred3.extend(sarima_forecast)

    y_forecasted3 = pred3

    forecast_steps = 4
    pred4 = []
    for i in range((int((len(y_to_test)+1)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred4.extend(sarima_forecast)

    y_forecasted4 = pred4[:-1]

    forecast_steps = 5
    pred5 = []
    for i in range((int(len(y_to_test)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred5.extend(sarima_forecast)

    y_forecasted5 = pred5


    forecast_steps = 6
    pred6 = []
    for i in range((int((len(y_to_test)+3)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred6.extend(sarima_forecast)

    y_forecasted6 = pred6[:-3]


    forecast_steps = 7
    pred7 = []
    for i in range((int((len(y_to_test)+5)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred7.extend(sarima_forecast)

    y_forecasted7 = pred7[:-5]


    forecast_steps = 8
    pred8 = []
    for i in range((int((len(y_to_test)+1)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred8.extend(sarima_forecast)

    y_forecasted8 = pred8[:-1]


    forecast_steps = 9
    pred9 = []
    for i in range((int(len(y_to_test)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred9.extend(sarima_forecast)

    y_forecasted9 = pred9


    forecast_steps = 10
    pred10 = []
    for i in range((int((len(y_to_test)+5)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred10.extend(sarima_forecast)

    y_forecasted10 = pred10[:-5]


    forecast_steps = 11
    pred11 = []
    for i in range((int((len(y_to_test)+8)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred11.extend(sarima_forecast)

    y_forecasted11 = pred11[:-8]


    forecast_steps = 12
    pred12 = []
    for i in range((int((len(y_to_test)+9)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred12.extend(sarima_forecast)

    y_forecasted12 = pred12[:-9]

    forecast_steps = 18
    pred18 = []
    for i in range((int((len(y_to_test)+9)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred18.extend(sarima_forecast)

    y_forecasted18 = pred12[:-9]

    forecast_steps = 24
    pred24 = []
    for i in range((int((len(y_to_test)+9)/forecast_steps))):
        sarima_fit = sm.tsa.SARIMAX(y_to_train.append(y_to_test[:i+forecast_steps]), order=order, seasonal_order=seasonal_order).fit()
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
        pred24.extend(sarima_forecast)

    y_forecasted24 = pred24[:-9]

    pred_dynamic = results.get_prediction(start=pred_date, dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean

    data_arima = {'y_forecasted': y_forecasted, 'y_forecasted1': y_forecasted1, 'y_forecasted2': y_forecasted2,
           'y_forecasted3': y_forecasted3, 'y_forecasted4': y_forecasted4, 'y_forecasted5': y_forecasted5,
           'y_forecasted6': y_forecasted6, 'y_forecasted7': y_forecasted7, 'y_forecasted8': y_forecasted8,
           'y_forecasted9': y_forecasted9, 'y_forecasted10': y_forecasted10, 'y_forecasted11': y_forecasted11,
           'y_forecasted12': y_forecasted12, 'y_forecasted18': y_forecasted18, 'y_forecasted24': y_forecasted24,
           'y_forecasted_dynamic': y_forecasted_dynamic}
    # Create a DataFrame
    data_pred_arima = pd.DataFrame(data_arima)

    writer = pd.ExcelWriter('output_arima.xlsx', engine='xlsxwriter')
    # Convert the DataFrame to an Excel object
    data_pred_arima.to_excel(writer, sheet_name='Sheet1')
    # Save the Excel file
    writer.save()
