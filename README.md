# New York City Department of Correction Jail Population Timeseries Analysis

This repository contains the scripts used to engineer and select features for the time series analysis and forecasting of DOC Jail Population.
After testing multiple machine learning techniques for predicting future jail population (ARIMA, ARIMAX, and linear regression) and measure the performance across all models, we chose to use a linear regression model. 

After analyzing the average daily jail population from July 2021 to current day, we found that the jail population does not have a strong seasonal component but does have a strong upward trend through time. We also found that the ADP is highly correlation with the admission counts for a given time period and the number of individuals in custody 4 months prior to the current month. This aligns with our subject matter expertise as we know that average length of stay for individuals in custody is roughly 110 days or 3.5 months. Therefore, we expect the number of individuals in custody 4 months ago to be negatively correlated with the current month's population. 

Explore the requirements, data sources, and methodology used throughout this project below. The finding in this repository are used to create the automated projections in the New York City Jail Population Tracker here [ADD LINK HERE].

# Requirements


# Data Sources

All of the data used in this analysis are open-source data provided by NYC Open Data Portal. We access this data through the Socrata querying language, SoSQL. 

**Dependent Variable/Target Variable: 30-day Average Daily Population of Individuals in DOC Custody.**

Published by the NYC Department of Correction (DOC) on the NYC Open Data Portal, the [Daily Inmates in Custody](https://data.cityofnewyork.us/Public-Safety/Daily-Inmates-In-Custody/7479-ugqb/about_data) dataset reports individuals in custody and their attributes (race, gender, age, custody level, mental health designation, top charge, legal status, sealed status, security risk group membership, and infraction flag).

It is important to note that this dataset is not historical, therefore we leveraged Amazon Web Services and the Socrata API querying functionality to create a database to store and aggregate the daily files. The earliest record we have obtained from the Open Data Portal is July 27, 2021 and we hope to continue data collection in the future. 

Please note that if you try to replicate this analysis and you are outside of the DCJ organization, you will be unable to access the aggregated data through the boto3 package due to lack of credentials.

**Inmate Admission & Discharges**

Published by the NYC Department of Correction (DOC) on the NYC Open Data Portal, the [Inmate Admissions](https://data.cityofnewyork.us/Public-Safety/Inmate-Admissions/6teu-xtgp/about_data) and [Inmate Discharges](https://data.cityofnewyork.us/Public-Safety/Inmate-Discharges/94ri-3ium/about_data) datasets report inmate admissions and discharges with attributes (race, gender, legal status, top charge). This data set excludes Sealed Cases. Unlike the Daily Inmates in Custody dataset, these datasets are historical so we can use the Socrata API querying functionality to get 30-day admission & discharge counts for the same 30-day periods as our daily population data mentioned above. We also using the Inmate Discharge dataset to calculate average length of stay for individuals discharged every 30-days. We will use these metrics as inputs to our model as admission and discharge counts, and length of stay are highly correlated to the population of individuals in DOC custody during a given time period.



# Methodology

**Step 1: Data Retreival and Aggregation**


**Step 2: Feature Engineering**

**Step 3: Feature Selection**

**Step 4: Model Training & Cross-Validation**