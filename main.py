
import util
from util import *
import sarima_processing
from sarima_processing import *
import transform_forecast
from transform_forecast import *
import evaluation_models
from evaluation_models import *

def main():
    """# Read Data"""

    # Read the data
    df_FPI = pd.read_csv('FPI.csv')
    df_cereals = pd.read_csv('cereals.csv')
    df_dairy = pd.read_csv('dairy.csv')
    df_oils = pd.read_csv('oils.csv')

    """## Plot Train / Test sample"""

    dataframes = [df_FPI, df_cereals, df_dairy, df_oils]
    
    util.plot_train_test_sample(dataframes, '1990-01-01', '2012-03-01', '2012-03-01', '2022-02-01')

    """# SARIMA Modeling"""

    d_FPI = df_FPI.load
    d_cereals = df_cereals.load
    d_dairy = df_dairy.load
    d_oils = df_oils.load

    """## Check stationary"""

    timeseries_FPI = df_FPI['load']
    timeseries_cereals = df_cereals['load']
    timeseries_dairy = df_dairy['load']
    timeseries_oils = df_oils['load']

    print(sarima_processing.check_stationarity(timeseries_FPI))
    print(sarima_processing.check_stationarity(timeseries_cereals))
    print(sarima_processing.check_stationarity(timeseries_dairy))
    print(sarima_processing.check_stationarity(timeseries_oils))

    data_list = [d_FPI, d_cereals, d_dairy, d_oils]  # List of data
    orders_list = [(1,1,1), (1,1,1), (1,1,1), (1,1,1)]  # List of orders
    seasonal_orders_list = [(0,1,1,12)] * 4  # List of seasonal orders

    # Call the function with your data and orders
    results = sarima_processing.fit_sarimax_models(data_list, orders_list, seasonal_orders_list)

    # Access the results for each variable
    results_FPI = results['FPI']
    results_cereals = results['cereals']
    results_dairy = results['dairy']
    results_oils = results['oils']

    predictions = sarima_processing.get_predictions_and_confidence_intervals(results)

    # Access the predictions and confidence intervals for each variable
    pred_FPI = predictions['pred_FPI']
    pred_ci_FPI = predictions['pred_ci_FPI']
    pred_cereals = predictions['pred_cereals']
    pred_ci_cereals = predictions['pred_ci_cereals']
    pred_dairy = predictions['pred_dairy']
    pred_ci_dairy = predictions['pred_ci_dairy']
    pred_oils = predictions['pred_oils']
    pred_ci_oils = predictions['pred_ci_oils']

    data_arima = {'arima_FPI': pred_FPI.predicted_mean ,
                'arima_cereals': pred_cereals.predicted_mean ,
                'arima_dairy': pred_dairy.predicted_mean ,
                'arima_oils': pred_oils.predicted_mean ,
            }
    # Create a DataFrame
    data_pred_arima = pd.DataFrame(data_arima)
    writer = pd.ExcelWriter('output_arima.xlsx', engine='xlsxwriter')
    # Convert the DataFrame to an Excel object
    data_pred_arima.to_excel(writer, sheet_name='Sheet1')
    # Save the Excel file
    writer.save()

    """## SARIMA diagnostic"""

    results_FPI.plot_diagnostics(figsize=(15, 12))
    plt.show()
    results_cereals.plot_diagnostics(figsize=(15, 12))
    plt.show()
    results_dairy.plot_diagnostics(figsize=(15, 12))
    plt.show()
    results_oils.plot_diagnostics(figsize=(15, 12))
    plt.show()

    """## Forecast h-head-Period (h-step)"""

    train_FPI = pd.DataFrame(d_FPI[:266])
    test_FPI = pd.DataFrame(d_FPI[266:])
    train_cereals = pd.DataFrame(d_cereals[:266])
    test_cereals = pd.DataFrame(d_cereals[266:])
    train_dairy = pd.DataFrame(d_dairy[:266])
    test_dairy = pd.DataFrame(d_dairy[266:])
    train_oils = pd.DataFrame(d_oils[:266])
    test_oils = pd.DataFrame(d_oils[266:])

    y= df_oils.load
    order= (1, 1, 1)
    seasonal_order = (0, 1, 1, 12)
    seasonal_period = 12
    pred_date = 266
    y_to_train = train_oils.load
    y_to_test = test_oils.load

    sarima_processing.sarima_article(y,order,seasonal_order,seasonal_period,pred_date,y_to_train, y_to_test)

    y= df_FPI.load
    order= (1, 1, 1)
    seasonal_order = (0, 1, 1, 12)
    seasonal_period = 12
    pred_date = 266
    y_to_train = train_FPI.load
    y_to_test = test_FPI.load

    sarima_processing.sarima_article(y,order,seasonal_order,seasonal_period,pred_date,y_to_train, y_to_test)

    y= df_cereals.load
    order= (1, 1, 1)
    seasonal_order = (0, 1, 1, 12)
    seasonal_period = 12
    pred_date = 266
    y_to_train = train_cereals.load
    y_to_test = test_cereals.load

    sarima_processing.sarima_article(y,order,seasonal_order,seasonal_period,pred_date,y_to_train, y_to_test)

    y= df_dairy.load
    order= (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    seasonal_period = 12
    pred_date = 266
    y_to_train = train_dairy.load
    y_to_test = test_dairy.load

    sarima_processing.sarima_article(y,order,seasonal_order,seasonal_period,pred_date,y_to_train, y_to_test)

    """# Transformer"""

    args_FPI = {
        "num_epoches": 150,
        "step_per_epoch": 3,
        "lr": 0.005,
        'Transformerheads': 2,
        'global_hidden_size': 19,
        'noise_hidden_size': 19,
        'n_factors': 19,
        "likelihood": "g",
        "seq_len": 27,
        "num_obs_to_train": 50 * 3,
        "num_results_to_sample": 1,
        "show_plot": False,
        "run_test": True,
        "standard_scaler": False,
        "log_scaler": False,
        "mean_scaler": True,
        "batch_size": 36,
        "sample_size": 377,
    }
    num_obs_to_train=50*3
    seq_len_FPI = 27

    args_Cereals = {
    "num_epoches":150,
    "step_per_epoch": 3,
    "lr":0.005,
    'Transformerheads': 2,
    'global_hidden_size': 19,
    'noise_hidden_size': 19,
    'n_factors': 19,
    "likelihood":"g",
    "seq_len":12,
    "num_obs_to_train":50*3,
    "num_results_to_sample":1,
    "show_plot":False,
    "run_test":True,
    "standard_scaler":False,
    "log_scaler":False,
    "mean_scaler":True,
    "batch_size":36,
    "sample_size":377,
    }

    num_obs_to_train=50*3
    seq_len_Cereals = 12

    args_Dairy = {
    "num_epoches":150,
    "step_per_epoch": 3,
    "lr":0.005,
    'Transformerheads': 2,
    'global_hidden_size': 19,
    'noise_hidden_size': 19,
    'n_factors': 19,
    "likelihood":"g",
    "seq_len":15,
    "num_obs_to_train":50*3,
    "num_results_to_sample":1,
    "show_plot":False,
    "run_test":True,
    "standard_scaler":False,
    "log_scaler":False,
    "mean_scaler":True,
    "batch_size":36,
    "sample_size":377,
    }

    num_obs_to_train=50*3
    seq_len_Dairy = 25

    args_Oils = {
    "num_epoches":150,
    "step_per_epoch": 3,
    "lr":0.005,
    'Transformerheads': 2,
    'global_hidden_size': 19,
    'noise_hidden_size': 19,
    'n_factors': 19,
    "likelihood":"g",
    "seq_len":26,
    "num_obs_to_train":50*3,
    "num_results_to_sample":1,
    "show_plot":False,
    "run_test":True,
    "standard_scaler":False,
    "log_scaler":False,
    "mean_scaler":True,
    "batch_size":36,
    "sample_size":377,
    }

    num_obs_to_train=50*3
    seq_len_Oils = 26

    seq = [seq_len_FPI, seq_len_Cereals, seq_len_Dairy, seq_len_Oils]
    parameters = [args_FPI, args_Cereals, args_Dairy, args_Oils]
    datasets = ["FPI.csv", "cereals.csv", "dairy.csv", "oils.csv"]

    result_data_transform  = transform_forecast.process_data(seq, parameters, datasets)

    grouped_data = {}
    # Define the groups
    groups = ["FPI.csv", "cereals.csv", "dairy.csv", "oils.csv"]

    # Create dictionaries for each group
    for group in groups:
        grouped_data[group] = {
            f"p50_list_{group}": result_data_transform[f"p50_list_{group}"],
            f"p90_list_{group}": result_data_transform[f"p90_list_{group}"],
            f"p10_list_{group}": result_data_transform[f"p10_list_{group}"],
            f"true_values_{group}": result_data_transform[f"true_values_{group}"],
            f"x_test_list_{group}": result_data_transform[f"x_test_list_{group}"]
        }

    # Print the grouped data
    for group, data in grouped_data.items():
        print(f"Group: {group}")
        print(data)

    FPI_R = result_data_transform['true_values_FPI.csv']
    FPI_F = result_data_transform['p50_list_FPI.csv']
    p50_list = result_data_transform['p50_list_FPI.csv']
    p10_list= result_data_transform['p10_list_FPI.csv']
    p90_list= result_data_transform['p90_list_FPI.csv']
    true_values= result_data_transform['true_values_FPI.csv']

    fig = plt.figure(1, figsize=(6,5))
    plt.plot(p50_list[0:130])
    plt.plot(true_values[0:130])
    plt.fill_between(x=np.arange(130), y1=p10_list[0:130], y2=p90_list[0:130], alpha=0.5)
    plt.title('FPI Prediction')
    plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper right")
    ymin, ymax = plt.ylim()
    plt.xlabel('Year')
    plt.ylabel('Index')
    plt.show()

    cereals_R = result_data_transform['true_values_cereals.csv']
    cereals_F = result_data_transform['p50_list_cereals.csv']
    p50_list = result_data_transform['p50_list_cereals.csv']
    p10_list= result_data_transform['p10_list_cereals.csv']
    p90_list= result_data_transform['p90_list_cereals.csv']
    true_values= result_data_transform['true_values_cereals.csv']

    fig = plt.figure(1, figsize=(6,5))
    plt.plot(p50_list[0:130])
    plt.plot(true_values[0:130])
    plt.fill_between(x=np.arange(130), y1=p10_list[0:130], y2=p90_list[0:130], alpha=0.5)
    plt.title('Cereals Prediction')
    plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper right")
    ymin, ymax = plt.ylim()
    plt.xlabel('Year')
    plt.ylabel('Index')
    plt.show()

    dairy_R = result_data_transform['true_values_dairy.csv']
    dairy_F = result_data_transform['p50_list_dairy.csv']
    p50_list = result_data_transform['p50_list_dairy.csv']
    p10_list= result_data_transform['p10_list_dairy.csv']
    p90_list= result_data_transform['p90_list_dairy.csv']
    true_values= result_data_transform['true_values_dairy.csv']

    fig = plt.figure(1, figsize=(6,5))
    plt.plot(p50_list[0:130])
    plt.plot(true_values[0:130])
    plt.fill_between(x=np.arange(130), y1=p10_list[0:130], y2=p90_list[0:130], alpha=0.5)
    plt.title('Dairy Prediction')
    plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper right")
    ymin, ymax = plt.ylim()
    plt.xlabel('Year')
    plt.ylabel('Index')
    plt.show()

    oils_R = result_data_transform['true_values_oils.csv']
    oils_F = result_data_transform['p50_list_oils.csv']
    p50_list = result_data_transform['p50_list_oils.csv']
    p10_list= result_data_transform['p10_list_oils.csv']
    p90_list= result_data_transform['p90_list_oils.csv']
    true_values= result_data_transform['true_values_oils.csv']

    fig = plt.figure(1, figsize=(6,5))
    plt.plot(p50_list[0:130])
    plt.plot(true_values[0:130])
    plt.fill_between(x=np.arange(130), y1=p10_list[0:130], y2=p90_list[0:130], alpha=0.5)
    plt.title('Oils Prediction')
    plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper right")
    ymin, ymax = plt.ylim()
    plt.xlabel('Year')
    plt.ylabel('Index')
    plt.show()

    data_forecast_transformer_FPI = {'fpi_r': FPI_R, 'fpi_f': FPI_F}
    data_forecast_transformer_cereals = {'cereals_r': cereals_R, 'cereals_f': cereals_F}
    data_forecast_transformer_dairy = {'dairy_r': dairy_R, 'dairy_f': dairy_F}
    data_forecast_transformer_oils = {'oils_r': oils_R, 'oils_f': oils_F}

    # Create a DataFrame
    data_pred_FPI = pd.DataFrame(data_forecast_transformer_FPI)
    data_pred_cereals = pd.DataFrame(data_forecast_transformer_cereals)
    data_pred_dairy = pd.DataFrame(data_forecast_transformer_dairy)
    data_pred_oils = pd.DataFrame(data_forecast_transformer_oils)

    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter('output_file.xlsx') as writer:
        # Write each DataFrame to a separate sheet
        data_pred_FPI.to_excel(writer, sheet_name='FPI')
        data_pred_cereals.to_excel(writer, sheet_name='Cereals')
        data_pred_dairy.to_excel(writer, sheet_name='Dairy')
        data_pred_oils.to_excel(writer, sheet_name='Oils')

    """# Evaluation models"""

    file_path = 'Rst_forecasting.xlsx'
    start_date = '1990-01-01'
    end_date = '2012-03-01'
    data_dict = evaluation_models.process_excel_file(file_path, start_date, end_date)
    forecast_data_dict = evaluation_models.generate_forecast_data_dict(data_dict)
    confidence_intervals = evaluation_models.generate_confidence_intervals_samples(forecast_data_dict)
    evaluation_models.plot_forecast_with_intervals_zahra(file_path, start_date, end_date, suffix='F')
    evaluation_models.plot_forecast_with_intervals_zahra(file_path, start_date, end_date, suffix='C')
    evaluation_models.plot_forecast_with_intervals_zahra(file_path, start_date, end_date, suffix='D')
    evaluation_models.plot_forecast_with_intervals_zahra(file_path, start_date, end_date, suffix='O')
    


if __name__ == "__main__":
    main()