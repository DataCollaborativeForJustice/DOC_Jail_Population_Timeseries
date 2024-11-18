# New York City Department of Correction Jail Population Time Series Analysis

This repository contains the scripts used to engineer and select features for the time series analysis and forecasting of the NYC Jail Population.

After testing multiple machine learning techniques for projecting the future jail population (ARIMA, ARIMAX, SARIMAX, and linear regression) and after measuring the performance across all models, we chose to use a SARIMAX model. 

After analyzing the average daily jail population (ADP) from June 2016 to current day, we found that the jail population does not have a strong seasonal component but does have a strong upward trend through time. We also found that the ADP is highly correlated with the misdemeanor, nonviolent felony, and violent felony crime and arrest counts for a given time 30-day period. 

Explore the data sources and methodology used throughout this project below. The findings in this repository are used to create the automated projections in the New York City Jail Population Tracker [here](https://nyc-jail-population-tracker.datacollaborativeforjustice.org/).


# Data Sources

All of the data used in this analysis are open-source data provided by NYC Open Data Portal. We access this data through the Socrata querying language, SoSQL. 

**Dependent Variable/Target Variable: Daily Jail Population**

Published by the NYC Department of Correction (DOC) on the NYC Open Data Portal,
the [Daily Inmates in Custody](https://data.cityofnewyork.us/Public-Safety/Daily-Inmates-In-Custody/7479-ugqb/about_data) dataset reports people in custody and their attributes (race, gender, age, custody level, mental health designation, top charge, legal status, sealed status, security risk group membership, and infraction flag).

This dataset is not historical. Each day’s jail population snapshot replaces the one posted for the previous day. Therefore we leveraged Amazon Web Services and the Socrata API querying
functionality to create a database to store and aggregate the daily files. The earliest record we have obtained from the Open Data Portal is June 2, 2016 and we will continue data collection in the future.

**Jail Admission & Discharges**

Published by the NYC Department of Correction (DOC) on the NYC Open Data Portal, [Admissions](https://data.cityofnewyork.us/Public-Safety/Inmate-Admissions/6teu-xtgp/about_data) and [Discharges](https://data.cityofnewyork.us/Public-Safety/Inmate-Discharges/94ri-3ium/about_data) datasets report people's admissions and discharges with attributes (race, gender, legal status, top charge). This dataset excludes sealed cases. Unlike the daily jail population dataset, these datasets are historical so we can use the Socrata API querying functionality to obtain 30-day admission & discharge counts for the same 30-day periods as our daily population data mentioned above. We also use the discharge dataset to calculate average length of stay for people discharged every 30 days. We will use these metrics as inputs to our model, given that admission and discharge counts and length of stay are highly correlated to the population in DOC custody during any given time period.

**Crime Complaints & Arrests**

Published by the New York City Police Department (NYPD) on the NYC Open Data Portal, [crime complaints](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data) and [arrest](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u/about_data) datasets
report valid felony, misdemeanor, and violation crimes reported to and arrests made by the NYPD beginning in 2006.
Unlike the daily jail population dataset, these datasets are historical. Therefore, we can use the Socrata API querying
functionality to obtain 30-day crime & arrest counts for the same prior 30-day periods as our daily population data mentioned above.


# Methodology

## Step 1: Data Retrieval and Aggregation

**Notebook:** `/Scripts/01_get_data.ipynb`

We will use boto3 package to get the historical daily inmates in custody data from our secure S3 bucket in AWS. *Note: you will not be able to retrieve this data unless you are a part of the DCJ organization and have configured the proper AWS credentials for your computer.*

**If you are attempting to clone this repository but do not have access to our AWS data warehouse, SKIP this notebook and use the static data in the `/Data` folder of this project.**

Independent/Exogenous Variables: We will also use the SoSQL language to grab exogenous variables from NYC Open Data Portal. Refer to the following user-defined functions (UDFs) in the functions.py file to learn more about how we query data from the portal:

* `get_agg_admit_dis_data`

* `get_crime_data`

* `get_arrest_data`

* `get_los_data`

Both datasets will then be aggregated to 30-day rolling averages or counts to be used in the analysis.

## Step 2: Descriptive and Exploratory Analysis

In this section, we explore the characteristics and underlying patterns of the average daily jail population and various exogenous variables. **The analysis below was conducted on population data from June 2, 2016 through September 29, 2024.**

### Dependent Variable: NYC Jail Population

**Notebook:** `/Scripts/02_descriptive_analysis.ipynb`

The goal of this noteboook is to visualize the past trends of the jail population data. This will inform major trends in the jail population data for later time series analysis and inform which exogenous variables are more or less correlated to our DV.

We utilize several techniques to understand the distribution and structure of the jail population data, starting with visualizations of its distribution. Time series decomposition is employed to break down the signal into its core components—trend, seasonality, and residuals—offering insights into the temporal dynamics of the jail population.

We also assess the strength of relationships between the jail population and exogenous variables using Pearson’s correlation, which measures the linear association between these variables. The exogenous variables we have collected include:

* 30-Day total misdemeanor crime, nonviolent felony crime, and violent felony crime counts.

* 30-Day total misdemeanor arrest, nonviolent felony arrest, and violent felony arrest counts.

* 30-Day DOC admission and discharge counts.

* 30-Day average length of stay (LOS).

Autocorrelation and partial autocorrelation plots of the historical ADP provide additional information about the persistence of values over time and help identify underlying autoregressive relationships. Some of the methods described below include bimodal distributions, k-means clustering, seasonal decomposition, ACF & PACF, Pearson’s correlation, and partial correlation.

### Exogenous Variables Time Series Analysis

**Notebook:** `/Scripts/03_exog_vars_ts_analysis.ipynb`

The exogenous variables we have collected for this analysis include several crime, arrest, and jail activity metrics sampled at 30-day intervals. These are:

* 30-Day total misdemeanor crime, nonviolent felony crime, and violent felony crime counts.

* 30-Day total misdemeanor arrest, nonviolent felony arrest, and violent felony arrest counts.

* 30-Day DOC admission and discharge counts.

* 30-Day average length of stay (LOS).

To gain a better understanding of the trends within each exogenous variable (misdemeanor, nonviolent & violent felony crime and arrest counts), we conducted the same descriptive and exploratory tests as mentioned above for the exogenous variables (seasonal decomposition, ADF tests, ACF and PACF).

## Step 3: Model Training & Cross-Validation

In this notebook we will create a method that iterates through the possible ARIMA parameters (p,d,q) and measures the average mean absolute error (MAE) across a number of train-test splits conducted using a [rolling-window technique](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4). Doing so allows us to pick the best model based on the out-of-sample performance and will ensure our future projections are as accurate as possible. 

After further investigation, not included in this notebook, we have decided to look into the following exogenous variables:

* 30-Day misdemeanor, violent felony, and nonviolent felony crime counts

* 30-Day misdemeanor, violent felony, and nonviolent felony arrest counts

* 30-Day Admissions to DOC.

Instead of feeding all of these variables into our final ARIMA model simultaneously, we will conduct a "funnel approach", which aims to replicate the way the criminal legal system operates. Specifically a crime occurs first, which may or may not be followed by an arrest, which may or may not be followed by detention (jail admission). This funnel approach will help us minimize collinearity throughout the modelling process and will help us dictate whether or not these exogenous variables are increasing our predictive power by being compared to a model with the absence of exogenous variables.

# Requirements & Dependencies

```
boto3==1.28.4
matplotlib==3.7.1
numpy==1.24.3
pandas==2.2.3
pingouin==0.5.5
pmdarima==2.0.3
scikit_learn==1.3.0
seaborn==0.13.2
statsmodels==0.14.0
```
