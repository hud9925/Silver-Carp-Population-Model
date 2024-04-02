import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import DBSCAN
from scipy.stats import norm
import seaborn as sns

kms_per_radian = 6371.0088

# Load the data
lake_superior_data = pd.read_csv("Lake Superior Water Quality 2000-present.csv")

# List of codes to keep
# codes_to_keep = [6, 100, 102, 104, 109, 445, 90]
codes_to_keep = [6, 100, 102, 104, 109, 245, 90]

# Filter the DataFrame and drop unnecessary columns
filtered_df = lake_superior_data[lake_superior_data['CODE'].isin(codes_to_keep)].drop(
    columns=["CRUISE_PLAN", "LAST_DATE_UPDATED", "SOUNDING", "DEPTH_TO", "SEQ_NO", "LATITUDE", "LONGITUDE", "PSN"]
)

# Ensure the sorting doesn't have inplace=True as it would return None
filtered_df_sorted = filtered_df.sort_values(by=['LATITUDE_DD', "LONGITUDE_DD"])

# Extract coordinates for clustering
coords = filtered_df_sorted[['LATITUDE_DD', 'LONGITUDE_DD']]

# Define epsilon (1 km) in radians for use in haversine formula
epsilon = 1 / kms_per_radian

# DBSCAN model
db = DBSCAN(eps=epsilon, min_samples=20, algorithm='ball_tree', metric='haversine')

# Fit the model and predict clusters
clusters = db.fit_predict(np.radians(coords))

# Add the cluster labels to the sorted DataFrame
filtered_df_sorted['cluster'] = clusters



# Save the DataFrame with clusters to a new CSV file
filtered_df_sorted.to_csv('filtered_data_with_clusters.csv', index=False)

# filter for water temperatures where column "VALUES" == "TEMPERATURE (OF WATER)", and dissolved oxygen FULL_NAME == "DISSOLVED OXYGEN, PROFILER WQP"
temperature_df = filtered_df_sorted[filtered_df_sorted['FULL_NAME'] == "TEMPERATURE (OF WATER)"]


# Ensure 'STN_DATE' is a datetime object
temperature_df['STN_DATE'] = pd.to_datetime(temperature_df['STN_DATE'])

# Filter for only July to September for all years
summer_temps_df = temperature_df[temperature_df['STN_DATE'].dt.month.isin([7, 8, 9])]

# Now, calculate the average, min, and max summer temperature for each cluster
avg_summer_temps_per_cluster = summer_temps_df.groupby('cluster')['VALUE'].agg(['mean', 'min', 'max']).reset_index()

# Rename the columns for clarity
avg_summer_temps_per_cluster.rename(columns={
    'mean': 'AVG_SUMMER_TEMP',
    'min': 'MIN_SUMMER_TEMP',
    'max': 'MAX_SUMMER_TEMP'
}, inplace=True)

# Filter for dissolved oxygen measurements(lowest dissolved oxygen would be during the summer months)
# oxygen_data = filtered_df_sorted[filtered_df_sorted['FULL_NAME'] == "DISSOLVED OXYGEN, PROFILER WQP"]
oxygen_data = filtered_df_sorted[filtered_df_sorted['FULL_NAME'] == "OXYGEN,CONCENTRATION DISSOLVED"]
oxygen_data['STN_DATE'] = pd.to_datetime(oxygen_data['STN_DATE'])

summer_oxygen_df = oxygen_data[oxygen_data['STN_DATE'].dt.month.isin([7,8,9])]

# Calculate statistics for dissolved oxygen
oxygen_stats = summer_oxygen_df.groupby('cluster')['VALUE'].agg(['mean', 'min', 'max']).reset_index()
oxygen_stats.rename(columns={'mean': 'AVG_OXYGEN', 'min': 'MIN_OXYGEN', 'max': 'MAX_OXYGEN'}, inplace=True)

# Save the DataFrame with average summer temperatures to a new CSV file
avg_summer_temps_per_cluster.to_csv('avg_summer_temps_per_cluster.csv', index=False)

# Merge temperature and dissolved oxygen statistics
combined_stats = pd.merge(avg_summer_temps_per_cluster, oxygen_stats, on='cluster', how='outer')

# Save to CSV
combined_stats.to_csv('combined_cluster_stats.csv', index=False)

# Calculate mean latitude and longitude for each cluster
cluster_coords = filtered_df_sorted.groupby('cluster')[['LATITUDE_DD', 'LONGITUDE_DD']].mean().reset_index()

# Now, merge this with your combined_stats DataFrame to include coordinates
combined_stats_with_coords = pd.merge(combined_stats, cluster_coords, on='cluster', how='left')

# Save to CSV
combined_stats_with_coords.to_csv('combined_cluster_stats_with_coords.csv', index=False)



# visualizing the data 
# # Boxplot for summer temperatures for each cluster
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='cluster', y='VALUE', data=summer_temps_df)
# plt.title('Summer Temperatures for Each Cluster')
# plt.xlabel('Cluster')
# plt.ylabel('Temperature (°C)')
# plt.xticks(rotation=45)  # Rotate x labels for better readability if needed
# plt.show()

# # If you prefer a violin plot, which also shows the kernel density estimation of the distribution:
# plt.figure(figsize=(12, 6))
# sns.violinplot(x='cluster', y='VALUE', data=summer_temps_df)
# plt.title('Summer Temperatures for Each Cluster')
# plt.xlabel('Cluster')
# plt.ylabel('Temperature (°C)')
# plt.xticks(rotation=45)  # Rotate x labels for better readability if needed
# plt.show()

# # Initialize lists to store the min and max temperatures for each cluster
# cluster_min_temps = []
# cluster_max_temps = []
# cluster_labels = []

# # Loop through each cluster
# for cluster in sorted(temperature_df['cluster'].unique()):
#     # Filter data for the current cluster
#     cluster_data = temperature_df[temperature_df['cluster'] == cluster]
#     # Calculate min and max temperatures
#     min_temp = cluster_data['VALUE'].min()
#     max_temp = cluster_data['VALUE'].max()
#     # Append to lists
#     cluster_min_temps.append(min_temp)
#     cluster_max_temps.append(max_temp)
#     # Save the cluster label
#     cluster_labels.append(f'Cluster {cluster}')
#     # Print the min and max temperatures
#     print(f'Cluster {cluster}: Min Temp = {min_temp}, Max Temp = {max_temp}')


# # Make sure the 'STN_DATE' column is in datetime format
# temperature_df['STN_DATE'] = pd.to_datetime(temperature_df['STN_DATE'])

# # Filter for rows where temperature is greater than or equal to 15 degrees
# warm_temps_df = temperature_df[(temperature_df['FULL_NAME'] == "TEMPERATURE (OF WATER)") & (temperature_df["VALUE"] >= 15)]

# # Extract year from 'STN_DATE'
# warm_temps_df['YEAR'] = warm_temps_df['STN_DATE'].dt.year

# # Group by cluster and year, and count unique dates
# warm_days_per_cluster_year = warm_temps_df.groupby(['cluster', 'YEAR'])['STN_DATE'].nunique().reset_index()

# # Rename the columns for clarity
# warm_days_per_cluster_year.rename(columns={'STN_DATE': 'NUM_WARM_DAYS'}, inplace=True)

# # save to a datagrame with the count of warm days to a csv file
# warm_days_per_cluster_year.to_csv("warm_days_per_cluster_year_superior.csv")








# # Now we'll plot the ranges as a horizontal bar chart
# fig, ax = plt.subplots()

# # The bar locations (y-coordinates)
# y_positions = range(len(cluster_labels))

# # The bar heights (temperature differences)
# bar_heights = [max_temp - min_temp for min_temp, max_temp in zip(cluster_min_temps, cluster_max_temps)]

# # Plot horizontal bars
# ax.barh(y_positions, bar_heights, left=cluster_min_temps, color='skyblue')

# # Set the y-ticks to be the cluster labels
# ax.set_yticks(y_positions)
# ax.set_yticklabels(cluster_labels)

# # Add labels and a title
# ax.set_xlabel('Temperature Range (°C)')
# ax.set_title('Temperature Ranges for Each Cluster')

# # Invert the y-axis so that the bottom bar is the first cluster
# ax.invert_yaxis()

# plt.show()





# # Create a GeoAxes in the PlateCarree projection.
# extent for Lake Superior (approximate coordinates)
# extent = [-92.5, -84.5, 46.5, 49.5]

# # Create a figure with an appropriate size
# # Increase the width of the figure to ensure space for the legend
# fig = plt.figure(figsize=(13, 10))  
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# ax.set_extent(extent)

# # Add features to the map
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAKES, alpha=0.5)
# ax.add_feature(cfeature.RIVERS)

# # Plot each cluster with a different color
# for cluster in filtered_df_sorted['cluster'].unique():
#     cluster_data = filtered_df_sorted[filtered_df_sorted['cluster'] == cluster]
#     ax.scatter(cluster_data['LONGITUDE_DD'], cluster_data['LATITUDE_DD'], s=10, label=f'Cluster {cluster}', transform=ccrs.Geodetic())

# plt.savefig('lake_superior_map.png', bbox_inches='tight')
# plt.close()

# items_per_column = 15 

# # Get the handles and labels from the original plot
# handles, labels = ax.get_legend_handles_labels()

#     # Calculate the number of required columns in the legend figure (round up division)
#     num_columns = int(np.ceil(len(labels) / items_per_column))

#     # Create a new figure for the legend with a number of subplots equal to the number of required columns
#     fig_leg, axs_leg = plt.subplots(1, num_columns, figsize=(2*num_columns, 8))  # Adjust figsize accordingly

#     # Remove axes for all subplots
#     for ax_leg in axs_leg:
#         ax_leg.axis('off')

#     # Split the handles and labels into groups for each subplot
#     for col in range(num_columns):
#         start = col * items_per_column
#         end = start + items_per_column
#         fig_leg.legend(handles[start:end], labels[start:end], loc='center left', bbox_to_anchor=(col, 0.5, 1.0, 1), frameon=False)

#     # Save the figure containing only the legends
#     plt.savefig('lake_superior_legend.png', bbox_inches='tight')
#     plt.close()  # Close the plot to avoid displaying it
