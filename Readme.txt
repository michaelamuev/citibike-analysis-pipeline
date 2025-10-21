NYC CitiBike Data Analysis and Weather Correlation Pipeline
Overview

This Python project performs a full ETL and analysis pipeline on NYC CitiBike datasets, integrating trip, station, and weather data from Kaggle.
It automates data acquisition, cleaning, organization, statistical analysis, and visualization—producing structured outputs, summary reports, and graphics on trip patterns, station activity, and weather impact.

The goal is to provide reproducible insights into CitiBike usage trends across years (2013–2017) and analyze how factors such as temperature affect ridership.

Key Features

Automated Kaggle Download
Downloads CitiBike trip data (2013–2017) and NYC weather data directly using the Kaggle API.

Data Cleaning and Validation
Handles duplicates, invalid coordinates, date formats, and outliers in both trip and weather datasets.

Structured Data Output
Saves cleaned and organized CSVs under directories by type (stations, trips, weather) and by year/month.

Statistical Summaries
Generates growth rates, user-type distributions, trip duration stats, and top stations.

Data Visualizations
Creates heatmaps, density plots, and bar charts showing system growth, monthly usage, and seasonal trends.

Weather Impact Analysis
Quantifies trip frequency below freezing temperatures and maps station activity during cold weather.

Project Structure
├── citibike_complete.py
├── raw_data/                  # Automatically downloaded Kaggle datasets
├── clean_data/                # Cleaned station, trip, and weather CSVs
├── organized_data/
│   ├── stations/
│   ├── trips/
│   └── weather/
├── statistical_analysis/
│   ├── statistical_summary.txt
│   ├── system_growth.png
│   ├── cold_weather_usage.png
│   └── cold_weather_station_activity.png
└── visualizations/
    ├── trip_density.png
    ├── monthly_trips_heatmap.png
    ├── user_type_by_year.png
    └── monthly_totals_heatmap.png

Main Functions
Function	Purpose
setup_kaggle()	Installs and authenticates Kaggle API credentials for automated dataset access.
download_citibike_data()	Downloads and extracts both CitiBike and NYC weather datasets into /raw_data.
clean_stations_data()	Validates station data (coordinates, capacity, booleans, IDs).
clean_trips_data()	Cleans and filters trip data, ensuring valid durations and station coordinates.
clean_weather_data()	Parses and formats temperature, precipitation, and snow columns.
organize_data()	Exports sorted and partitioned CSVs by month and region for reproducibility.
perform_statistical_analysis()	Computes trip trends, user distributions, and writes summaries.
create_visualizations()	Generates growth, usage, and density charts using Matplotlib and Seaborn.
analyze_cold_weather_usage()	Evaluates ridership on sub-freezing days, producing visual reports.
create_seasonal_analysis()	Builds a heatmap of monthly trip volumes over multiple years.
main()	Orchestrates the full pipeline end-to-end.
Requirements

Python 3.9+

Libraries:

pip install pandas numpy matplotlib seaborn scipy requests


Kaggle API Setup

Go to your Kaggle Account Settings
.

Under “API,” click Create New API Token.

Place the downloaded kaggle.json file in either your working directory or ~/.kaggle/.

How to Run
python citibike_complete.py


If the Kaggle API token is missing, the script will display instructions on how to add it before proceeding.

Outputs

After execution, the project will:

Create organized folders with cleaned data by month and region.

Generate analytical summaries in /statistical_analysis.

Produce heatmaps and plots under /visualizations.

Write a detailed statistical summary to statistical_summary.txt.

Future Enhancements

Add command-line arguments to specify year or region filters

Introduce multiprocessing for large datasets

Incorporate GeoPandas and DBSCAN for geographic clustering of stations

Build an interactive Streamlit dashboard to explore results

Author

Michael Amuev
MBA Candidate, Baruch College – Data Analytics and Programming