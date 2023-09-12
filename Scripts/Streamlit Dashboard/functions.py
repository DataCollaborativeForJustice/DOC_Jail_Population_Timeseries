import pandas as pd
import urllib.request, json
import requests
import urllib.parse


def get_data(url):
    """ 
    This function retrieves the data from the NYC Open data portal for the 
    admission and discharge datasets published monthly by DOC.

    Input - 
    url: the socrata API url found on the dataset's page

    Output - 
    df: all the data associated with the url saved in a df structure.
    """
    query = (url+'?'
            "$select=*"
            "&$limit=500000")
    query = query.replace(" ", "%20")
    response = urllib.request.urlopen(query)
    data = json.loads(response.read())
    
    #store in dataframe
    df = pd.DataFrame(data,columns = data[0].keys())
    
    #define month and year columns and specify data type
    df['admitted_dt'] = pd.to_datetime(df['admitted_dt'])
    df['discharged_dt'] = pd.to_datetime(df['discharged_dt'])

    df['admitted_mo'] = df['admitted_dt'].dt.month
    df['discharged_mo'] = df['discharged_dt'].dt.month

    df['admitted_yr'] = df['admitted_dt'].dt.year
    df['discharged_yr'] = df['discharged_dt'].dt.year
    
    df['admitted_dt'] =  df['admitted_dt'].dt.date
    df['discharged_dt'] =  df['discharged_dt'].dt.date

    return df