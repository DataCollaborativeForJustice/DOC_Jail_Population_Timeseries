#import packages
import pandas as pd
import matplotlib.pyplot as plt
import boto3
import urllib.request
import json
from io import StringIO
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm



def check_file_existence(bucket_name,folder_name,file_name):
    """
    Function checks whether or not a specified file exists in its 
    designated location. Returns True if the file exists, otherwise False. 
    """
    try:
        s3_client = boto3.client('s3')
        s3_client.head_object(Bucket=bucket_name, Key=folder_name + file_name)
        return True
    except:
        return False

def get_file(bucket_name,folder_name,file_name): 
    """
    Function checks whether or not a specified file exists in its 
    designated location. If the file exists, the function retreives it from the 
    designated s3 bucket. 
    """
    if check_file_existence(bucket_name,folder_name,file_name):
        try:
            #get the merged file and save as df
            print('Trying to get object from bucket')
            s3_client = boto3.client('s3')
            file_content = s3_client.get_object(Bucket = bucket_name, Key = folder_name + file_name)['Body'].read().decode('utf-8')
            print('Got file contents from AWS S3')
            file_df = pd.read_csv(StringIO(file_content))
            print('Saved file contents as dataframe')
            return file_df
        except Exception as e:
            return e
        
#test functions for the streamlit dashboard
def get_agg_admit_dis_data(first_st_date_adp, api_url, date_col):
    """
    first_st_date_adp: datetime.date
        The date of the first start date 30-day period in the interval data
    api_url: str
        URL of Socrata API endpoint of interest
    date_col: str
        Either date_col or 'DISCHARGED_DT' depending which dataset you are querying from
    """
    if date_col == 'ADMITTED_DT':
        count_col = 'admission_count'

    elif date_col == 'DISCHARGED_DT':
        count_col = 'discharge_count'

    elif date_col != 'ADMITTED_DT' and date_col != 'DISCHARGED_DT':
        print('Date column is not correct and query will not work. Please enter the correct string, either ADMITTED_DT or DISCHARGED_DT.')

    # Define the SQL query separately
    sql_query = ("SELECT "
                f"date_trunc_ymd({date_col}) as {date_col}, "
                f"count(distinct INMATEID) as {count_col} "
                f"WHERE {date_col} >= '{first_st_date_adp}' "
                f"GROUP BY {date_col} "
                "LIMIT 10000")

    # Encode SQL query for URL
    encoded_query = urllib.parse.quote(sql_query)

    # Construct the full URL query
    final_query = f'{api_url}?$query={encoded_query}'
    # Send the request and load the response data
    response = urllib.request.urlopen(final_query)
    data = json.loads(response.read())

    # Store in dataframe
    df = pd.DataFrame(data, columns=data[0].keys())
    #specify data types
    df[date_col] = df[date_col].astype('datetime64[ns]')
    df[count_col] = df[count_col].astype(int)
    #define max and min dates for future calculations
    max_date = df[date_col].max()
    min_date = df[date_col].min()
    #aggregate to 30 day intervals
    # Resample the DataFrame to 30-day intervals
    interval_data = df.resample('30D', on=date_col, origin= min_date, closed='left', label='left').agg({count_col: 'sum'}).fillna(0).reset_index()
    # Rename columns
    interval_data = interval_data.rename(columns={date_col: 'Start Date', count_col: count_col})
    # Calculate the Start Date column
    interval_data['End Date'] = interval_data['Start Date'] + pd.to_timedelta(29, unit='D')
    #add date related regressors
    interval_data['Month'] = interval_data['Start Date'].dt.month
    interval_data['Year'] = interval_data['Start Date'].dt.year
    #calculate the days between start period and last date in admission df
    interval_data['Days to Max Date'] = (max_date - interval_data['Start Date']).dt.days
    # Display just the date portion of the start/end date columns and localize to specific timezone
    interval_data['Start Date'] = interval_data['Start Date'].dt.tz_localize('America/New_York').dt.date
    interval_data['End Date'] = interval_data['End Date'].dt.tz_localize('America/New_York').dt.date
    
    #make sure the last data point has a reasonable admission count
    #if the period has less than 30 days of admission data, we will use the avg of the last two data points
    #as the adjusted admission count for the given time period

    adj_count = []
    for i, row in interval_data.iterrows():
        if row['Days to Max Date'] < 30:
            ma2 = (interval_data.iloc[i-1,interval_data.columns.get_loc(count_col)] +
                        interval_data.iloc[i-2,interval_data.columns.get_loc(count_col)]) / 2
            adj_count.append(round(ma2))
        else:
            adj_count.append(row[count_col])
    #clean up to column name for final dataframe
    final_col_name = ' '.join([word.capitalize() for word in count_col.split('_')])
    interval_data['Adjusted '+final_col_name] = adj_count
    
    _30_day_admit_dis_df = interval_data.drop(columns = count_col)
    
    return _30_day_admit_dis_df


#test functions for the streamlit dashboard
def get_los_data(first_st_date_adp):
    """
    first_st_date_adp: datetime.date
        The date of the first start date 30-day period in the interval data
    """
    dis_url = 'https://data.cityofnewyork.us/resource/94ri-3ium.json'

    dis_query = ("SELECT "
                "distinct INMATEID, "
                "ADMITTED_DT, "
                "DISCHARGED_DT "
                f"WHERE DISCHARGED_DT >= '{first_st_date_adp}' "
                f"GROUP BY INMATEID, ADMITTED_DT, DISCHARGED_DT "
                "LIMIT 1000000")

    # Encode SQL query for URL
    encoded_query = urllib.parse.quote(dis_query)

    # Construct the full URL query
    final_dis_query = f'{dis_url}?$query={encoded_query}'

    # Send the request and load the response data
    response = urllib.request.urlopen(final_dis_query)
    data = json.loads(response.read())

    # Store in dataframe
    dis_df = pd.DataFrame(data, columns=data[0].keys())
    
    #specify data types before join
    dis_df[['ADMITTED_DT','DISCHARGED_DT']] = dis_df[['ADMITTED_DT','DISCHARGED_DT']].astype('datetime64[ns]')
    dis_df['INMATEID'] = dis_df['INMATEID'].astype(int)

    dis_df['LOS'] = (dis_df['DISCHARGED_DT'] - dis_df['ADMITTED_DT']).dt.days
    
    #define max and min dates for future calculations
    max_date = dis_df['DISCHARGED_DT'].max()
    min_date = dis_df['DISCHARGED_DT'].min()
    #aggregate to 30 day intervals
    # Resample the DataFrame to 30-day intervals
    interval_data = dis_df.resample('30D', on='DISCHARGED_DT', origin= min_date, closed='left', label='left').agg({'LOS': 'mean'}).fillna(0).reset_index()
    # Rename columns
    interval_data = interval_data.rename(columns={'DISCHARGED_DT': 'Start Date', 'LOS': 'Avg LOS Days'})
    # Calculate the Start Date column
    interval_data['End Date'] = interval_data['Start Date'] + pd.to_timedelta(29, unit='D')
    #add date related regressors
    interval_data['Discharge Month'] = interval_data['Start Date'].dt.month
    interval_data['Discharge Year'] = interval_data['Start Date'].dt.year
    #calculate the days between start period and last date in admission df
    interval_data['Days to Max Date'] = (max_date - interval_data['Start Date']).dt.days
    # Display just the date portion of the start/end date columns and localize to specific timezone
    interval_data['Start Date'] = interval_data['Start Date'].dt.tz_localize('America/New_York').dt.date
    interval_data['End Date'] = interval_data['End Date'].dt.tz_localize('America/New_York').dt.date
    
    #make sure the last data point has a reasonable admission count
    #if the period has less than 30 days of admission data, we will use the avg of the last two data points
    #as the adjusted admission count for the given time period

    adj_count = []
    for i, row in interval_data.iterrows():
        if row['Days to Max Date'] < 30:
            ma2 = (interval_data.iloc[i-1,interval_data.columns.get_loc('Avg LOS Days')] +
                        interval_data.iloc[i-2,interval_data.columns.get_loc('Avg LOS Days')]) / 2
            adj_count.append(round(ma2))
        else:
            adj_count.append(round(row['Avg LOS Days']))
    
    interval_data['Adjusted Avg LOS Days'] = adj_count
    
    _30_day_los_df = interval_data.drop(columns = ['Avg LOS Days','Days to Max Date'])
    
    return _30_day_los_df

def train_test_MLR(source_df,target_variable, regressor_ls, test_size = 0.2, random_state = None):
    # Scale the entire dataset
    sc = MinMaxScaler()
    data = source_df[[target_variable] + regressor_ls]
    data_sc = sc.fit_transform(data)

    # Convert the array to a DataFrame
    data_sc = pd.DataFrame(data=data_sc, columns=data.columns)

    # Add constant to the dataset
    data_sc = sm.add_constant(data_sc, prepend=False, has_constant='add')

    # Split the dataset into features (X) and target (y)
    X = data_sc.drop(columns=[target_variable])
    y = data_sc[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)
    #train and test
    # Fit the linear regression model on the entire dataset
    model = sm.OLS(y_train, X_train).fit()

    # Make in sample predictions
    IS_pred = model.predict(X_train)
    # Make out of  sample predictions
    OS_pred = model.predict(X_test)

    #inverse scale the model inputs to get IS predicts
    IS_inputs = pd.concat([IS_pred,X_train.drop(columns=['const'])],axis = 1)
    IS_inputs = IS_inputs.rename(columns={0:target_variable})
    IS_inputs = sc.inverse_transform(IS_inputs)
    #do the same with OS data
    OS_inputs = pd.concat([OS_pred,X_test.drop(columns=['const'])],axis = 1)
    OS_inputs = OS_inputs.rename(columns={0:target_variable})
    OS_inputs = sc.inverse_transform(OS_inputs)
    #accuracy in terms of non-scaled values
    IS_mse = mean_squared_error(data[target_variable].iloc[y_train.index], IS_inputs[:,0])
    IS_mae = mean_absolute_error(data[target_variable].iloc[y_train.index], IS_inputs[:,0])

    # Make out of  sample predictions
    OS_mse = mean_squared_error(data[target_variable].iloc[y_test.index], OS_inputs[:,0])
    OS_mae = mean_absolute_error(data[target_variable].iloc[y_test.index], OS_inputs[:,0])

    print(IS_mse, IS_mae)
    print(OS_mse, OS_mae)
    print(model.summary())


    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('Linear Regression Model Training')

    actual, = plt.plot(data.index, data[target_variable], 'bo-', label='Actual Data')
    predicted_IS = plt.scatter(y_train.index, IS_inputs[:, 0], color='green', label='Predicted In Sample')
    predicted_OS = plt.scatter(y_test.index, OS_inputs[:, 0], color='red', label='Predicted Out of Sample')

    plt.legend(handles=[actual, predicted_IS, predicted_OS])
    plt.show()

    return model, IS_mae, OS_mae

def fit_scale_linear_reg(final_df,regressor_ls):
    # Scale the entire dataset
    sc = MinMaxScaler()
    data = final_df[['ADP'] + regressor_ls]
    data_sc = sc.fit_transform(data)

    # Convert the array to a DataFrame
    data_sc = pd.DataFrame(data=data_sc, columns=data.columns)

    # Add constant to the dataset
    data_sc = sm.add_constant(data_sc, prepend=False, has_constant='add')

    # Split the dataset into features (X) and target (y)
    X = data_sc.drop(columns=['ADP'])
    y = data_sc['ADP']

    #train test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    #train and test
    # Fit the linear regression model on the entire dataset
    model = sm.OLS(y_train, X_train).fit()

    # Make in sample predictions
    IS_pred = model.predict(X_train)
    # Make out of  sample predictions
    OS_pred = model.predict(X_test)

    #inverse scale the model inputs to get IS predicts
    IS_inputs = pd.concat([IS_pred,X_train.drop(columns=['const'])],axis = 1)
    IS_inputs = IS_inputs.rename(columns={0:'ADP'})
    IS_inputs = sc.inverse_transform(IS_inputs)
    #do the same with OS data
    OS_inputs = pd.concat([OS_pred,X_test.drop(columns=['const'])],axis = 1)
    OS_inputs = OS_inputs.rename(columns={0:'ADP'})
    OS_inputs = sc.inverse_transform(OS_inputs)
    #accuracy in terms of non-scaled values
    IS_mae = mean_absolute_error(data.iloc[:train_size]['ADP'], IS_inputs[:,0])

    # Make out of  sample predictions
    OS_mae = mean_absolute_error(data.iloc[train_size:]['ADP'], OS_inputs[:,0])

    print('The in sample MAE:', IS_mae)
    print('The out of sample MAE:', OS_mae)
    print(model.summary())

    fig = plt.figure(figsize=(10,5))
    fig.suptitle(f'Linear Regression Model Training')

    actual, = plt.plot(data.index,data['ADP'], 'bo-', label='Actual Data')
    predicted_IS, = plt.plot(data.iloc[:train_size].index, IS_inputs[:,0], 'go-', label='Predicted In Sample')
    predicted_OS, = plt.plot(data.iloc[train_size:].index, OS_inputs[:,0], 'ro-', label='Predicted Out of Sample')

    plt.legend(handles=[actual,predicted_IS,predicted_OS])
    plt.show()

    return model, IS_mae, OS_mae