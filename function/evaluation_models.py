import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def process_excel_file(file_path, start_date, end_date):
    """
    Process an Excel file containing time-series data.

    Parameters:
    - file_path (str): Path to the Excel file.
    - start_date (str): Start date for data analysis.
    - end_date (str): End date for data analysis.

    Returns:
    - data_dict (dict): A dictionary containing processed data.
    """
    # Read data from Excel file
    sheet_names = ['F', 'C', 'D', 'O']
    dfs = {}
    for sheet_name in sheet_names:
        dfs[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)

    # Generate date ranges
    date_tr = pd.date_range(start=start_date, end=end_date, freq='M')
    date_ts = {
        'F': pd.date_range(start='2012-03-01', end='2022-02-01', freq='M'),
        'C': pd.date_range(start='2012-03-01', end='2022-11-01', freq='M'),
        'D': pd.date_range(start='2012-03-01', end='2023-05-01', freq='M'),
        'O': pd.date_range(start='2012-03-01', end='2021-09-01', freq='M')
    }

    # Create dataframes for time series data
    ts_data = {}
    for sheet_name in sheet_names:
        ts_data[sheet_name] = pd.DataFrame({'date': date_ts[sheet_name], 'value': dfs[sheet_name].iloc[:, 1]})
        ts_data[sheet_name].set_index('date', inplace=True)

    # Create a dictionary to store the data
    data_dict = {
        'dfs': dfs,
        'date_tr': date_tr,
        'date_ts': date_ts,
        'ts_data': ts_data
    }

    return data_dict



def generate_forecast_data_dict(data_dict):
    """
    Generate forecast data from the provided data dictionary.

    Parameters:
    - data_dict (dict): A dictionary containing data necessary for generating forecasts.
        It should have the following keys:
            - 'date_ts': A dictionary containing date ranges for different time series.
            - 'dfs': A dictionary containing dataframes for different time series.

    Returns:
    - forecast_data (dict): A dictionary containing forecasted dataframes.
        Keys are constructed based on forecast steps and time series identifiers.
        Each value is a DataFrame with dates as index and corresponding forecasted values.
    """
    date_ts = data_dict['date_ts']
    dfs = data_dict['dfs']

    forecast_data = {}

    for key in dfs.keys():
        # Step 1
        forecast_data[f'f_FPI_{key}'] = pd.DataFrame({'date': date_ts[key], 'value': dfs[key]['forecast_transformers']})
        forecast_data[f'f_FPI_{key}'].set_index('date', inplace=True)

        # Step 6
        forecast_data[f'fdyn_FPI6_{key}'] = pd.DataFrame({'date': date_ts[key], 'value': dfs[key]['forecast_arima_6']})
        forecast_data[f'fdyn_FPI6_{key}'].set_index('date', inplace=True)

        # Step 12
        forecast_data[f'fdyn_FPI12_{key}'] = pd.DataFrame({'date': date_ts[key], 'value': dfs[key]['forecast_arima_12']})
        forecast_data[f'fdyn_FPI12_{key}'].set_index('date', inplace=True)

        # Step 24
        forecast_data[f'fdyn_FPI24_{key}'] = pd.DataFrame({'date': date_ts[key], 'value': dfs[key]['forecast_arima_24']})
        forecast_data[f'fdyn_FPI24_{key}'].set_index('date', inplace=True)

        # Dynamic Step
        forecast_data[f'fdyn_FPI_{key}'] = pd.DataFrame({'date': date_ts[key], 'value': dfs[key]['forecast_arima_Dyn']})
        forecast_data[f'fdyn_FPI_{key}'].set_index('date', inplace=True)

    return forecast_data




def generate_confidence_intervals_samples(forecast_data_dict, num_samples=50):
    """
    Generate confidence intervals for the forecasted data using bootstrapping.

    Parameters:
    - forecast_data_dict (dict): A dictionary containing forecasted dataframes.
        Keys represent different forecasted variables.
        Values are DataFrames with dates as index and forecasted values.
    - num_samples (int): The number of bootstrap samples to generate. Default is 50.

    Returns:
    - confidence_intervals (dict): A dictionary containing upper and lower bounds of confidence intervals.
        Keys are constructed based on forecasted variables and indicate upper or lower bounds.
        Values are arrays representing the upper or lower bounds for each time step in the forecast.
    """
    z_score = 1.96  # Z-score for 95% confidence interval

    # Extract forecast data from the forecast_data_dict
    f_FPI = forecast_data_dict['f_FPI_F']
    f_Cereals = forecast_data_dict['f_FPI_C']
    f_Dairy = forecast_data_dict['f_FPI_D']
    f_Oils = forecast_data_dict['f_FPI_O']
    fdyn_FPI = forecast_data_dict['fdyn_FPI_F']
    fdyn_Cereals = forecast_data_dict['fdyn_FPI_C']
    fdyn_Dairy = forecast_data_dict['fdyn_FPI_D']
    fdyn_Oils = forecast_data_dict['fdyn_FPI_O']
    fdyn_FPI6 = forecast_data_dict['fdyn_FPI6_F']
    fdyn_Cereals6 = forecast_data_dict['fdyn_FPI6_C']
    fdyn_Dairy6 = forecast_data_dict['fdyn_FPI6_D']
    fdyn_Oils6 = forecast_data_dict['fdyn_FPI6_O']
    fdyn_FPI12 = forecast_data_dict['fdyn_FPI12_F']
    fdyn_Cereals12 = forecast_data_dict['fdyn_FPI12_C']
    fdyn_Dairy12 = forecast_data_dict['fdyn_FPI12_D']
    fdyn_Oils12 = forecast_data_dict['fdyn_FPI12_O']
    fdyn_FPI24 = forecast_data_dict['fdyn_FPI24_F']
    fdyn_Cereals24 = forecast_data_dict['fdyn_FPI24_C']
    fdyn_Dairy24 = forecast_data_dict['fdyn_FPI24_D']
    fdyn_Oils24 = forecast_data_dict['fdyn_FPI24_O']

    # Initialize dictionaries to store upper and lower bounds
    confidence_intervals = {}

    # Generate confidence intervals for each variable
    for variable_name in forecast_data_dict.keys():
        # Generate samples
        samples = np.random.normal(loc=forecast_data_dict[variable_name].value, scale=1.0, size=(num_samples, len(forecast_data_dict[variable_name].value)))

        # Calculate mean and standard deviation of samples
        mean_samples = np.mean(samples, axis=0)
        std_samples = np.std(samples, axis=0)

        # Calculate upper and lower bounds
        upper_bound = mean_samples + z_score * std_samples
        lower_bound = mean_samples - z_score * std_samples

        # Store upper and lower bounds in the confidence_intervals dictionary
        confidence_intervals[f'upper_bound_{variable_name}'] = upper_bound
        confidence_intervals[f'lower_bound_{variable_name}'] = lower_bound

    return confidence_intervals



def plot_forecast_with_intervals_zahra(file_path, start_date, end_date, suffix='F'):
    """
    Generate a plot displaying forecasted values with confidence intervals.

    Parameters:
    - file_path (str): Path to the Excel file containing data.
    - start_date (str): Start date for data analysis.
    - end_date (str): End date for data analysis.
    - suffix (str): Suffix to identify the specific time series. Default is 'F'.

    Returns:
    - None

    This function processes the provided Excel file, generates forecast data using
    different models, calculates confidence intervals, and plots the forecasted
    values along with their confidence intervals. The plots are saved as a JPEG image
    named 'forecast_FPI.jpeg' and displayed.
    """
    # Step 1: Process the Excel file
    data_dict = process_excel_file(file_path, start_date, end_date)

    # Step 2: Generate forecast data dictionary
    forecast_data_dict = generate_forecast_data_dict(data_dict)

    # Step 3: Generate confidence intervals
    confidence_intervals = generate_confidence_intervals_samples(forecast_data_dict)

    # Unpack the forecast data and confidence intervals
    f_FPI = forecast_data_dict[f'f_FPI_{suffix}']
    fdyn_FPI = forecast_data_dict[f'fdyn_FPI_{suffix}']
    fdyn_FPI6 = forecast_data_dict[f'fdyn_FPI6_{suffix}']
    fdyn_FPI12 = forecast_data_dict[f'fdyn_FPI12_{suffix}']
    fdyn_FPI24 = forecast_data_dict[f'fdyn_FPI24_{suffix}']

    upper_bound_f = confidence_intervals[f'upper_bound_f_FPI_{suffix}']
    lower_bound_f = confidence_intervals[f'lower_bound_f_FPI_{suffix}']
    upper_bound_fdyn6 = confidence_intervals[f'upper_bound_fdyn_FPI6_{suffix}']
    lower_bound_fdyn6 = confidence_intervals[f'lower_bound_fdyn_FPI6_{suffix}']
    upper_bound_fdyn12 = confidence_intervals[f'upper_bound_fdyn_FPI12_{suffix}']
    lower_bound_fdyn12 = confidence_intervals[f'lower_bound_fdyn_FPI12_{suffix}']
    upper_bound_fdyn24 = confidence_intervals[f'upper_bound_fdyn_FPI24_{suffix}']
    lower_bound_fdyn24 = confidence_intervals[f'lower_bound_fdyn_FPI24_{suffix}']

    # Create the plot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Plotting for each subplot
    axes[0, 0].plot(data_dict['ts_data'][suffix].index, data_dict['ts_data'][suffix].value, label="Actual", color='black')
    axes[0, 0].plot(f_FPI.index, f_FPI.value, label="Forecast Transformers", color='red', ls='-.')
    axes[0, 0].fill_between(f_FPI.index, lower_bound_f, upper_bound_f, alpha=0.3, label='95% CI Transformers')
    axes[0, 0].plot(fdyn_FPI.index, fdyn_FPI.value, label="Forecast SARIMA", color='blue', ls=':')
    axes[0, 0].fill_between(fdyn_FPI.index, lower_bound_f, upper_bound_f, alpha=0.3, label='95% CI SARIMA')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title('Dynamic')

    axes[0, 1].plot(data_dict['ts_data'][suffix].index, data_dict['ts_data'][suffix].value, label="Actual", color='black')
    axes[0, 1].plot(f_FPI.index, f_FPI.value, label="Forecast Transformers", color='red', ls='-.')
    axes[0, 1].fill_between(f_FPI.index, lower_bound_f, upper_bound_f, alpha=0.3, label='95% CI Transformers')
    axes[0, 1].plot(fdyn_FPI6.index, fdyn_FPI6.value, label="Forecast SARIMA", color='blue', ls=':')
    axes[0, 1].fill_between(fdyn_FPI6.index, lower_bound_fdyn6, upper_bound_fdyn6, alpha=0.3, label='95% CI SARIMA')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title('6-head-steps')

    axes[1, 0].plot(data_dict['ts_data'][suffix].index, data_dict['ts_data'][suffix].value, label="Actual", color='black')
    axes[1, 0].plot(f_FPI.index, f_FPI.value, label="Forecast Transformers", color='red', ls='-.')
    axes[1, 0].fill_between(f_FPI.index, lower_bound_f, upper_bound_f, alpha=0.3, label='95% CI Transformers')
    axes[1, 0].plot(fdyn_FPI12.index, fdyn_FPI12.value, label="Forecast SARIMA", color='blue', ls=':')
    axes[1, 0].fill_between(fdyn_FPI12.index, lower_bound_fdyn12, upper_bound_fdyn12, alpha=0.3, label='95% CI SARIMA')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title('12-head-steps')

    axes[1, 1].plot(data_dict['ts_data'][suffix].index, data_dict['ts_data'][suffix].value, label="Actual", color='black')
    axes[1, 1].plot(f_FPI.index, f_FPI.value, label="Forecast Transformers", color='red', ls='-.')
    axes[1, 1].fill_between(f_FPI.index, lower_bound_f, upper_bound_f, alpha=0.3, label='95% CI Transformers')
    axes[1, 1].plot(fdyn_FPI24.index, fdyn_FPI24.value, label="Forecast SARIMA", color='blue', ls=':')
    axes[1, 1].fill_between(fdyn_FPI24.index, lower_bound_fdyn24, upper_bound_fdyn24, alpha=0.3, label='95% CI SARIMA')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title('24-head-steps')

    # Adjust layout to avoid overlapping
    plt.tight_layout()
    # Save the plot as a JPEG image
    plt.savefig('forecast_FPI.jpeg', format='jpeg', dpi=300)
    # Display the plots
    plt.show()
