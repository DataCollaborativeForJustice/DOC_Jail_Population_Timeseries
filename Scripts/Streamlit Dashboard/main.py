import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import datetime
from functions import get_data

#define socrata urls from NYC open data portal
admit_url = 'https://data.cityofnewyork.us/resource/6teu-xtgp.json'
dis_url = 'https://data.cityofnewyork.us/resource/94ri-3ium.json'
#Step 1: get data
admissions = get_data(admit_url)
discharges = get_data(dis_url)

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


monthly_counts.head()

#get lastest month's daily population change
currentMonth = datetime.datetime.now().month
currentYear = datetime.datetime.now().year

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

# # Create an interactive time series plot
# st.header('Monthly Jail Population Change Time Series')
# st.line_chart(monthly_counts.set_index('Year-Mo')['Population Change'])

# # Allow users to hover over data points to see additional information
# st.write("Hover over the plot to see more details.")

# # Optionally, you can add tooltips for Admissions and Discharges
# hover_data = {'Population Change': True, 'Admissions': monthly_counts['Admissions'], 'Discharges': monthly_counts['Discharges']}
# st.line_chart(monthly_counts.set_index('Year-Mo'), use_container_width=True, chart_data=hover_data)

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
# Display the Streamlit app

