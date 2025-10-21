# Imports and dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import zipfile
import json
import requests
from pathlib import Path
import subprocess
from scipy import stats
import shutil
# import geopandas as gpd  # Commented out until we implement geographic analysis
# from sklearn.cluster import DBSCAN  # For future station clustering analysis

# Data acquisition and API setup
def setup_kaggle():
    """Set up Kaggle API access similar to Colab notebook approach."""
    try:
        # Install kaggle
        subprocess.check_call(['pip', 'install', '-q', 'kaggle'])
        
        # Create .kaggle directory in user home
        kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
        os.makedirs(kaggle_dir, exist_ok=True)
        
        # Copy kaggle.json if it exists in current directory
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        local_json = 'kaggle.json'
        
        if os.path.exists(local_json):
            shutil.copy2(local_json, kaggle_json)
            os.chmod(kaggle_json, 0o600)
            return True
            
        if not os.path.exists(kaggle_json):
            print("\nKaggle API token not found!")
            print("Please follow these steps:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Scroll to 'API' section and click 'Create New API Token'")
            print("3. Download and place kaggle.json in:", os.getcwd())
            print("   or in:", kaggle_dir)
            return False
            
        return True
    except Exception as e:
        print(f"Error during Kaggle setup: {e}")
        return False

def download_citibike_data(output_dir='raw_data'):
    """Download Citibike and weather datasets from Kaggle using direct commands."""
    try:
        if not setup_kaggle():
            return False
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Download Citibike data
        print("\nDownloading Citibike dataset...")
        subprocess.check_call(['kaggle', 'datasets', 'download', '-d', 'fatihb/citibike-sampled-data-2013-2017'])
        citibike_zip = 'citibike-sampled-data-2013-2017.zip'
        
        # Download Weather data
        print("\nDownloading Weather dataset...")
        subprocess.check_call(['kaggle', 'datasets', 'download', '-d', 'mathijs/weather-data-in-new-york-city-2016'])
        weather_zip = 'weather-data-in-new-york-city-2016.zip'
        
        extracted_files = {}
        
        # Extract Citibike data
        print("\nExtracting Citibike data...")
        with zipfile.ZipFile(citibike_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            extracted_files['citibike'] = [f for f in zip_ref.namelist() if any(pattern in f.lower() for pattern in ['citibike-stations', 'citibike-trips'])]
        os.remove(citibike_zip)
        
        # Extract Weather data
        print("\nExtracting Weather data...")
        with zipfile.ZipFile(weather_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            extracted_files['weather'] = [f for f in zip_ref.namelist() if f.lower().endswith('.csv')]
        os.remove(weather_zip)
        
        return extracted_files
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False

def clean_stations_data(df):
    """Clean and validate Citibike station data.
    
    TODO: Add validation for station capacity vs historical max bikes
    FIXME: Some station coordinates seem off by ~0.01 degrees
    """
    df = df.copy()
    
    # Basic cleaning
    df = df.drop_duplicates(subset=['station_id_int'], keep='first')
    df['name'] = df['name'].fillna('Unknown Station')
    
    # Coordinate validation
    # bounds = {'lat': (40.6, 40.9), 'lon': (-74.1, -73.8)}  # NYC bounds
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Capacity validation
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
    df = df[df['capacity'] > 0]
    
    # Convert boolean columns
    bool_cols = ['eightd_has_key_dispenser', 'is_installed', 'is_renting', 'is_returning', 'eightd_has_available_keys']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Convert numeric columns
    int_cols = ['station_id_int', 'num_bikes_available', 'num_bikes_disabled', 'num_docks_available', 'num_docks_disabled']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df.reset_index(drop=True)

def clean_trips_data(df):
    df = df.copy()
    datetime_columns = ['starttime', 'stoptime']
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df = df.dropna(subset=datetime_columns)
    df['tripduration'] = pd.to_numeric(df['tripduration'], errors='coerce')
    df['trip_duration_minutes'] = df['tripduration'] / 60
    df = df[(df['trip_duration_minutes'] >= 1) & (df['trip_duration_minutes'] <= 24 * 60)]
    
    station_cols = ['start_station_latitude', 'start_station_longitude', 'end_station_latitude', 'end_station_longitude']
    for col in station_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=station_cols)
    df['usertype'] = df['usertype'].str.lower().str.strip()
    df['gender'] = df['gender'].map({0: 'unknown', 1: 'male', 2: 'female'}).fillna('unknown')
    
    df['birth_year'] = pd.to_numeric(df['birth_year'], errors='coerce')
    current_year = datetime.now().year
    df = df[(df['birth_year'].isna()) | ((current_year - df['birth_year'] >= 16) & (current_year - df['birth_year'] <= 90))]
    
    df['bikeid'] = pd.to_numeric(df['bikeid'], errors='coerce').fillna(0).astype(int)
    df = df.reset_index(drop=True)
    return df

def clean_weather_data(weather_df):
    weather_df = weather_df.copy()
    
    # Convert date column
    try:
        # First ensure date is string type
        weather_df['date'] = weather_df['date'].astype(str)
        # Try to parse dates with flexible parsing
        weather_df['date'] = pd.to_datetime(weather_df['date'], format='mixed', dayfirst=True)
    except Exception as e:
        print(f"Warning: Error parsing dates in weather data: {e}")
        return pd.DataFrame()
    
    # Handle precipitation
    if 'precipitation' in weather_df.columns:
        weather_df['precipitation'] = weather_df['precipitation'].replace('T', '0.001')
        weather_df['precipitation'] = pd.to_numeric(weather_df['precipitation'], errors='coerce')
    
    # Handle snow measurements
    for col in ['snow fall', 'snow depth']:
        if col in weather_df.columns:
            weather_df[col] = weather_df[col].replace('T', '0.1')
            weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
    
    # Temperature columns are already numeric
    temp_columns = ['average temperature', 'maximum temperature', 'minimum temperature']
    for col in temp_columns:
        if col in weather_df.columns:
            weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
    
    return weather_df

# Data organization and file management
def organize_data(stations_df, trips_df, weather_df):
    directories = ['raw_data', 'clean_data', 'organized_data/stations', 'organized_data/trips', 'organized_data/weather', 'visualizations']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    stations_df.to_csv('clean_data/clean_citibike_stations.csv', index=False)
    trips_df.to_csv('clean_data/clean_citibike_trips.csv', index=False)
    
    if not weather_df.empty:
        weather_df.to_csv('clean_data/clean_weather.csv', index=False)
    
    stations_df_sorted = stations_df.sort_values(['region_id', 'capacity', 'station_id_int'], ascending=[True, False, True])
    stations_df_sorted.to_csv('organized_data/stations/stations_sorted.csv', index=False)
    
    for region in stations_df['region_id'].unique():
        region_df = stations_df[stations_df['region_id'] == region]
        region_df.to_csv(f'organized_data/stations/region_{region}_stations.csv', index=False)
    
    active_stations = stations_df[stations_df['is_installed'] & stations_df['is_renting'] & stations_df['is_returning']]
    inactive_stations = stations_df[~(stations_df['is_installed'] & stations_df['is_renting'] & stations_df['is_returning'])]
    
    active_stations.to_csv('organized_data/stations/active_stations.csv', index=False)
    inactive_stations.to_csv('organized_data/stations/inactive_stations.csv', index=False)
    
    trips_df['year_month'] = trips_df['starttime'].dt.to_period('M')
    for period in trips_df['year_month'].unique():
        period_dir = f'organized_data/trips/{period}'
        os.makedirs(period_dir, exist_ok=True)
        period_data = trips_df[trips_df['year_month'] == period]
        period_data_sorted = period_data.sort_values(['start_station_id', 'starttime'])
        period_data_sorted.to_csv(f'{period_dir}/trips_{period}.csv', index=False)
    
    if not weather_df.empty and 'date' in weather_df.columns:
        try:
            weather_df['year_month'] = pd.to_datetime(weather_df['date']).dt.to_period('M')
            for period in weather_df['year_month'].unique():
                period_dir = f'organized_data/weather/{period}'
                os.makedirs(period_dir, exist_ok=True)
                period_data = weather_df[weather_df['year_month'] == period]
                period_data_sorted = period_data.sort_values('date')
                period_data_sorted.to_csv(f'{period_dir}/weather_{period}.csv', index=False)
            
            weather_summary = weather_df.groupby('year_month').agg({
                'precipitation': 'sum',
                'snow fall': 'sum',
                'snow depth': 'mean',
                'maximum temperature': ['max', 'mean'],
                'minimum temperature': ['min', 'mean'],
                'average temperature': 'mean'
            }).round(2)
            
            weather_summary.to_csv('organized_data/weather/weather_summary.csv')
        except Exception as e:
            print(f"\nWarning: Could not process weather data: {e}")

# Statistical analysis and reporting
def perform_statistical_analysis(trips, stations, weather):
    stats_dir = 'statistical_analysis'
    os.makedirs(stats_dir, exist_ok=True)
    
    trips['year'] = pd.to_datetime(trips['starttime']).dt.year
    trips['month'] = pd.to_datetime(trips['starttime']).dt.month
    
    # Year-over-year analysis
    yearly_counts = trips.groupby(['year', 'usertype']).size().unstack(fill_value=0)
    total_yearly = yearly_counts.sum(axis=1)
    growth_rate = total_yearly.pct_change() * 100
    yearly_pct = yearly_counts.div(yearly_counts.sum(axis=1), axis=0) * 100
    
    with open(os.path.join(stats_dir, 'statistical_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=== CitiBike System Analysis (2013-2017) ===\n\n")
        
        # Year-over-year statistics
        f.write("Annual Trip Summary:\n")
        f.write("-" * 50 + "\n")
        f.write(str(yearly_counts) + "\n\n")
        f.write(f"Total growth 2013-2017: {((total_yearly.iloc[-1]/total_yearly.iloc[0])-1)*100:.1f}%\n")
        f.write(f"Highest growth: {growth_rate.max():.1f}% ({growth_rate.idxmax()})\n\n")
        
        # Trip duration statistics
        f.write("\nTrip Duration Statistics (minutes):\n")
        f.write("-" * 50 + "\n")
        duration_stats = trips['trip_duration_minutes'].describe()
        f.write(str(duration_stats) + "\n\n")
        
        # Station popularity
        f.write("\nTop 5 Most Popular Start Stations:\n")
        f.write("-" * 50 + "\n")
        popular_stations = trips.groupby('start_station_name').size().sort_values(ascending=False).head()
        f.write(str(popular_stations) + "\n\n")
        
        # Station capacity statistics
        f.write("\nStation Capacity Statistics:\n")
        f.write("-" * 50 + "\n")
        capacity_stats = stations['capacity'].describe()
        f.write(str(capacity_stats) + "\n\n")
        
        # Top 10 stations by capacity
        f.write("\nTop 10 Stations by Capacity:\n")
        f.write("-" * 50 + "\n")
        top_capacity = stations.nlargest(10, 'capacity')[['name', 'capacity']]
        f.write(str(top_capacity) + "\n\n")
        
        # User type distribution
        f.write("\nUser Type Distribution:\n")
        f.write("-" * 50 + "\n")
        user_dist = trips['usertype'].value_counts()
        f.write(str(user_dist) + "\n\n")
    
    # Key visualizations
    plt.figure(figsize=(12, 8))
    ax = yearly_counts.plot(kind='bar', stacked=True, width=0.8)
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fmt='{:,.0f}')
    plt.title('CitiBike Usage Growth')
    plt.xlabel('Year')
    plt.ylabel('Number of Trips')
    plt.legend(title='User Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'system_growth.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    popular_stations.plot(kind='bar')
    plt.title('Top 5 Most Popular Start Stations')
    plt.xlabel('Station')
    plt.ylabel('Number of Trips')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'popular_stations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=stations, x='capacity', bins=30)
    plt.title('Station Capacity Distribution')
    plt.xlabel('Capacity')
    plt.ylabel('Count')
    plt.savefig(os.path.join(stats_dir, 'station_capacity_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Process cold weather data if available
    if not weather.empty:
        trips['date'] = pd.to_datetime(trips['starttime']).dt.date
        weather['date'] = weather['date'].dt.date
        cold_weather_data = trips.merge(weather, on='date', how='left')
        return yearly_counts, cold_weather_data
    else:
        return yearly_counts, pd.DataFrame()

def create_seasonal_analysis(trips):
    stats_dir = 'statistical_analysis'
    os.makedirs(stats_dir, exist_ok=True)
    
    # Create a month name mapping
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    trips['month'] = pd.to_datetime(trips['starttime']).dt.month
    trips['year'] = pd.to_datetime(trips['starttime']).dt.year
    
    # Convert month numbers to names before grouping
    trips['month_name'] = trips['month'].map(month_names)
    monthly_trips = trips.groupby(['year', 'month_name']).size().reset_index(name='trip_count')
    pivot_data = monthly_trips.pivot(index='month_name', columns='year', values='trip_count')
    
    # Sort the index by month number to maintain chronological order
    pivot_data = pivot_data.reindex([month_names[i] for i in range(1, 13)])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt=',')
    plt.title('Monthly Trip Counts by Year')
    plt.xlabel('Year')
    plt.ylabel('Month')
    plt.savefig(os.path.join(stats_dir, 'seasonal_heatmap.png'))
    plt.close()
    
    # Save with month names
    pivot_data.to_csv(os.path.join(stats_dir, 'monthly_trip_counts.csv'))

# Data visualization
def create_visualizations(trips_df, stations_df):
    os.makedirs('visualizations', exist_ok=True)
    plt.style.use('default')
    
    # Trip density visualization
    plt.figure(figsize=(12, 8))
    start_stations = trips_df.merge(stations_df, left_on='start_station_id', right_on='station_id_int', how='left')
    start_stations = start_stations.dropna(subset=['longitude', 'latitude'])
    
    hist, x_edges, y_edges = np.histogram2d(
        start_stations['longitude'],
        start_stations['latitude'],
        bins=50
    )
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    
    plt.scatter(start_stations['longitude'], start_stations['latitude'], c='blue', alpha=0.2, s=3)
    plt.contourf(X, Y, hist.T, levels=20, cmap='YlOrRd', alpha=0.5)
    plt.title('Citibike Trip Density')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Trip Density')
    plt.xlim(-74.02, -73.95)
    plt.ylim(40.68, 40.78)
    plt.savefig('visualizations/trip_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # User type distribution by year
    trips_df['year'] = trips_df['starttime'].dt.year
    user_type_by_year = pd.pivot_table(
        trips_df,
        values='tripduration',
        index='year',
        columns='usertype',
        aggfunc='count'
    )
    
    plt.figure(figsize=(10, 6))
    ax = user_type_by_year.plot(kind='bar', stacked=True)
    plt.title('Year-over-Year User Type Distribution')
    plt.xlabel('Year')
    plt.ylabel('Number of Trips')
    plt.legend(title='User Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f', label_type='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/user_type_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Monthly trip distribution heatmap
    trips_df['month'] = trips_df['starttime'].dt.strftime('%b')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Detailed monthly heatmap by year
    heatmap_data = trips_df.groupby(['year', 'month']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data[month_order]
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='g', linewidths=0.5)
    plt.title('Total CitiBike Trips per Month')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig('visualizations/monthly_trips_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Aggregated monthly totals
    month_totals = trips_df['month'].value_counts().reindex(month_order)
    plt.figure(figsize=(8, 2))
    sns.heatmap(month_totals.values.reshape(1, -1), annot=True, fmt='g',
                cmap='coolwarm', xticklabels=month_order, yticklabels=['Total Trips'],
                linewidths=0.5, annot_kws={'size': 8}, cbar=False)
    plt.title('Total CitiBike Trips by Month')
    plt.tight_layout()
    plt.savefig('visualizations/monthly_totals_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# Weather impact analysis
def analyze_cold_weather_usage(trips, weather, stations_df):
    stats_dir = 'statistical_analysis'
    os.makedirs(stats_dir, exist_ok=True)
    
    weather = clean_weather_data(weather)
    if weather.empty:
        print("Warning: Could not process weather data. Skipping cold weather analysis.")
        return
        
    trips['date'] = pd.to_datetime(trips['starttime']).dt.date
    weather['date'] = weather['date'].dt.date
    cold_weather_data = trips.merge(weather, on='date', how='left')
    
    with open(os.path.join(stats_dir, 'statistical_summary.txt'), 'a', encoding='utf-8') as f:
        f.write("\nCold Weather Analysis:\n")
        if 'average temperature' in cold_weather_data.columns:
            freezing_trips = cold_weather_data[cold_weather_data['average temperature'] <= 32]
            f.write(f"Number of trips in freezing weather: {len(freezing_trips):,}\n")
            f.write(f"Percentage of total trips: {len(freezing_trips)/len(trips)*100:.2f}%\n")
            
            # Temperature distribution histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(data=cold_weather_data, x='average temperature', hue='usertype', multiple="stack")
            plt.axvline(x=32, color='r', linestyle='--', label='Freezing Point')
            plt.title('Trip Distribution by Temperature')
            plt.xlabel('Temperature (°F)')
            plt.ylabel('Number of Trips')
            plt.legend(title='User Type')
            plt.savefig(os.path.join(stats_dir, 'cold_weather_usage.png'))
            plt.close()
            
            # Cold weather station activity map
            cold_days = weather[weather['average temperature'] < 32]['date']
            cold_trips_df = trips[trips['starttime'].dt.date.isin(cold_days)]
            
            station_trip_counts = cold_trips_df['start_station_id'].value_counts().reset_index()
            station_trip_counts.columns = ['station_id_int', 'trip_count']
            
            heatmap_data = pd.merge(
                station_trip_counts,
                stations_df[['station_id_int', 'latitude', 'longitude']],
                on='station_id_int',
                how='left'
            ).dropna(subset=['latitude', 'longitude'])
            
            plt.figure(figsize=(15, 13))
            scatter = plt.scatter(
                heatmap_data['longitude'],
                heatmap_data['latitude'],
                s=heatmap_data['trip_count'] * 9,
                c=heatmap_data['trip_count'],
                cmap='Reds',
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            plt.title('Citi Bike Stations with higher activity at Temp < 32°F', fontsize=14)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)
            plt.colorbar(scatter, label='Trip Count')
            plt.tight_layout()
            plt.savefig(os.path.join(stats_dir, 'cold_weather_station_activity.png'), dpi=300, bbox_inches='tight')
            plt.close()

# Main execution
def main():
    """Main execution function for Citibike data analysis pipeline.
    
    TODO: 
    - Add command line arguments for data ranges
    - Implement multiprocessing for large datasets
    - Add progress bars for long operations
    """
    try:
        # Create directory structure
        directories = ['raw_data', 'clean_data', 'organized_data/stations', 
                      'organized_data/trips', 'organized_data/weather', 'visualizations', 
                      'statistical_analysis']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Download and load data
        # DEBUG: Print file sizes to check for truncation
        # for f in os.listdir('raw_data'):
        #     print(f"{f}: {os.path.getsize(os.path.join('raw_data', f))} bytes")
        
        extracted_files = download_citibike_data('raw_data')
        if not extracted_files:
            print("Error downloading data. Proceeding with existing data if available.")
        
        stations_df = pd.read_csv('raw_data/citibike-stations.csv')
        trips_df = pd.read_csv('raw_data/citibike-trips.csv')
        weather_df = pd.read_csv('raw_data/weather_data_nyc_centralpark_2016(1).csv')
        
        stations_df = clean_stations_data(stations_df)
        trips_df = clean_trips_data(trips_df)
        weather_df = clean_weather_data(weather_df)
        
        organize_data(stations_df, trips_df, weather_df)
        yearly_counts, daily_data = perform_statistical_analysis(trips_df, stations_df, weather_df)
        create_seasonal_analysis(trips_df)
        analyze_cold_weather_usage(trips_df, weather_df, stations_df)
        create_visualizations(trips_df, stations_df)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        raise

if __name__ == "__main__":
    main() 