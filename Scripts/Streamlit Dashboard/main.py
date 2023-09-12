import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import datetime
import os
import plotly.express as px
import plotly.graph_objects as go


#set working directory
if os.getcwd() != "C:\\Users\\emjoh\\OneDrive\\Documents\\DCJ Analyses\\Jail_Population_Timeseries\\Scripts\\Streamlit Dashboard":
    os.chdir(os.getcwd() +"\\Jail_Population_Timeseries\\Scripts\\Streamlit Dashboard")

#import functions
from functions import get_data

#define socrata urls from NYC open data portal
admit_url = 'https://data.cityofnewyork.us/resource/6teu-xtgp.json'
dis_url = 'https://data.cityofnewyork.us/resource/94ri-3ium.json'

currentMonth = datetime.datetime.now().month
currentYear = datetime.datetime.now().year

while True:
    try:
        # Try to get the latest dataset; if it doesn't exist, let's pull the data from the portal
        latest_data = pd.read_csv('monthly_rikers_population_change.csv')
        admissions = pd.read_csv('DOC_admissions.csv')
        discharges = pd.read_csv('DOC_discharges.csv')
        print('Files exist')
        # If this file exists, check the last month & year
        last_yr_mo = latest_data.sort_values(by=['year', 'month']).tail(1)['Year-Mo'].values[0]

        if last_yr_mo == (datetime.datetime.today().replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%#m"):
            # If the file exists and the last data point is current, break the loop
            print('Files are up to date')
            monthly_counts = latest_data.copy()
            break
        else:
            print('Files are NOT up to date, please wait while I grab the newest data.')
            # If the last data point is not current, continue to the next iteration of the loop
            pass
    except Exception as e:
        # If the try statement raised an error, i.e., there is no file called latest data, print the error
        print(e)
        #Step 1: get data if we anticipate a new file from the portal, otherwise grab the previously saved files

        admissions = get_data(admit_url)
        print('Successfully downloaded admission data')
        discharges = get_data(dis_url)
        print('Successfully downloaded discharge data')

        #Step 2: Aggregate to monthly admit/discharge
        monthly_admits = admissions[['admitted_yr','admitted_mo','inmateid']].groupby(by = ['admitted_yr','admitted_mo']).nunique().reset_index()
        monthly_admits = monthly_admits.rename(columns = {'admitted_yr':'year',
                                                        'admitted_mo':'month',
                                                        'inmateid':'Admissions'})
        monthly_discharges = discharges[['discharged_yr','discharged_mo','inmateid']].groupby(by = ['discharged_yr','discharged_mo']).nunique().reset_index()
        monthly_discharges = monthly_discharges.rename(columns = {'discharged_yr':'year',
                                                                'discharged_mo':'month',
                                                                'inmateid':'Discharges'})

        #merge
        monthly_counts = monthly_admits.merge(monthly_discharges, on = ['year','month'])
        monthly_counts['Population Change'] = monthly_counts['Admissions'] - monthly_counts['Discharges']
        monthly_counts['Year-Mo'] = ['-'.join(i) for i in zip(monthly_counts["year"].map(str),monthly_counts["month"].map(str))]
        print('Successfully merged data')
        #save
        monthly_counts.to_csv('monthly_rikers_population_change.csv')
        admissions.to_csv('DOC_admissions.csv')
        discharges.to_csv('DOC_discharges.csv')
        print('Successfully saved data')
        monthly_counts.head()
        break

#get lastest month's daily population change
last_mo_admissions = admissions[(admissions['admitted_yr'] == currentYear) & (admissions['admitted_mo'] == currentMonth-1)]
last_mo_discharges = discharges[(discharges['discharged_yr'] == currentYear) & (discharges['discharged_mo'] == currentMonth-1)]

latest_admits = last_mo_admissions[['admitted_dt','inmateid']].groupby(by = 'admitted_dt').nunique().reset_index()
latest_discharge = last_mo_discharges[['discharged_dt','inmateid']].groupby(by = 'discharged_dt').nunique().reset_index()

#merge
latest_counts = latest_admits.merge(latest_discharge, left_on = 'admitted_dt',right_on = 'discharged_dt',suffixes=('_adm', '_dis'))
latest_counts['Population Change'] = latest_counts['inmateid_adm'] - latest_counts['inmateid_dis']

latest_daily_avg = round(latest_counts['Population Change'].mean(),2)
latest_admissions = monthly_counts[monthly_counts['Year-Mo'] == str(currentYear)+"-"+str(currentMonth-1)]['Admissions'].values[0]
latest_discharges = monthly_counts[monthly_counts['Year-Mo'] == str(currentYear)+"-"+str(currentMonth-1)]['Discharges'].values[0]

# Create Streamlit app
st.title('Riker\'s Island Monthly Population Analysis')

# Display the three print statements as text blocks
st.header('Population Change Analysis')
st.write(f"Last month the population in Riker's facilities changed by {latest_daily_avg} per day.")
st.write(f"Last month {latest_admissions} people were admitted to a Riker's facility.")
st.write(f"Last month {latest_discharges} people were discharged from a Riker's facility.")

# Create an interactive time series plot with 'o' markers and hover data using Altair
st.header('Monthly Jail Population Change Time Series')
# Use Altair to create the chart
chart = alt.Chart(monthly_counts).mark_line().encode(
    x='Year-Mo',
    y='Population Change',
    tooltip=['Year-Mo', 'Population Change', 'Admissions', 'Discharges']
).properties(
    width=800,
    height=400
) + alt.Chart(monthly_counts).mark_circle().encode(
    x='Year-Mo',
    y='Population Change',
    tooltip=['Year-Mo', 'Population Change', 'Admissions', 'Discharges']
).interactive()

st.altair_chart(chart, use_container_width=True)

# Create a histogram of monthly population change with 50 bins using Matplotlib
st.header('Histogram of Monthly Population Change')

# Add a slider for filtering data by 'Year-Mo'
selected_year_month = st.select_slider('Select Year-Mo', options=monthly_counts['Year-Mo'].unique())

# Filter the data based on the selected 'Year-Mo'
filtered_data = monthly_counts[monthly_counts['Year-Mo'] >= selected_year_month]


# Create an interactive histogram using Plotly for filtered data
fig = go.Figure()

# Create a histogram for the entire dataset
fig.add_trace(go.Histogram(
    x=monthly_counts['Population Change'],
    name='Entire Population Change Distribution',
    histnorm='probability',  # Normalize to probability
    opacity=0.5,  # Adjust opacity for desired visibility
    marker=dict(color='grey')
))

# Create a histogram for the filtered data
fig.add_trace(go.Histogram(
    x=filtered_data['Population Change'],
    name='Population Change distribution between {} and {}'.format(
        selected_year_month,
        monthly_counts.sort_values(by=['year', 'month']).tail(1)['Year-Mo'].values[0]
    ),
    histnorm='probability',  # Normalize to probability
    opacity=0.7,  # Adjust opacity for desired visibility
    marker=dict(color='blue')
))

# Set fixed x-axis limits
fig.update_xaxes(range=[-1200, 1200])  # Adjust the range as per your preference

# Set the legend labels
fig.update_layout(legend_title_text='Legend')
# Stack bars on top of each other
fig.update_layout(barmode='overlay')
st.plotly_chart(fig)
