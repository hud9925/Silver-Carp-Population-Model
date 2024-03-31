
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import DBSCAN


kms_per_radian = 6371.0088

# Load the data
lake_superior_data = pd.read_csv("Lake Superior Water Quality 2000-present.csv")

# List of codes to keep
codes_to_keep = [6, 100, 102, 104, 109, 445, 90]

# Filter the DataFrame and drop unnecessary columns
filtered_df = lake_superior_data[lake_superior_data['CODE'].isin(codes_to_keep)].drop(
    columns=["CRUISE_PLAN", "LAST_DATE_UPDATED", "SOUNDING", "DEPTH_TO", "SEQ_NO", "LATITUDE", "LONGITUDE", "PSN"]
)

# Ensure the sorting doesn't have inplace=True as it would return None
filtered_df_sorted = filtered_df.sort_values(by=['LATITUDE_DD', "LONGITUDE_DD"])

# Extract coordinates for clustering
coords = filtered_df_sorted[['LATITUDE_DD', 'LONGITUDE_DD']]

# Define epsilon (1 km) in radians for use in haversine formula
epsilon = 2 / kms_per_radian

# Create the DBSCAN model
db = DBSCAN(eps=epsilon, min_samples=20, algorithm='ball_tree', metric='haversine')

# Fit the model and predict clusters
clusters = db.fit_predict(np.radians(coords))

# Add the cluster labels to the sorted DataFrame
filtered_df_sorted['cluster'] = clusters

# Save the DataFrame with clusters to a new CSV file
filtered_df_sorted.to_csv('filtered_data_with_clusters.csv', index=False)

# extent for Lake Superior (approximate coordinates)
extent = [-92.5, -84.5, 46.5, 49.5]

# Create a figure with an appropriate size
# Increase the width of the figure to ensure space for the legend
fig = plt.figure(figsize=(13, 10))  

# Create a GeoAxes in the PlateCarree projection.
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(extent)

# Add features to the map
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Plot each cluster with a different color
for cluster in filtered_df_sorted['cluster'].unique():
    cluster_data = filtered_df_sorted[filtered_df_sorted['cluster'] == cluster]
    ax.scatter(cluster_data['LONGITUDE_DD'], cluster_data['LATITUDE_DD'], s=10, label=f'Cluster {cluster}', transform=ccrs.Geodetic())

plt.savefig('lake_superior_map.png', bbox_inches='tight')
plt.close()

items_per_column = 15 

# Get the handles and labels from the original plot
handles, labels = ax.get_legend_handles_labels()

# Calculate the number of required columns in the legend figure (round up division)
num_columns = int(np.ceil(len(labels) / items_per_column))

# Create a new figure for the legend with a number of subplots equal to the number of required columns
fig_leg, axs_leg = plt.subplots(1, num_columns, figsize=(2*num_columns, 8))  # Adjust figsize accordingly

# Remove axes for all subplots
for ax_leg in axs_leg:
    ax_leg.axis('off')

# Split the handles and labels into groups for each subplot
for col in range(num_columns):
    start = col * items_per_column
    end = start + items_per_column
    fig_leg.legend(handles[start:end], labels[start:end], loc='center left', bbox_to_anchor=(col, 0.5, 1.0, 1), frameon=False)

# Save the figure containing only the legends
plt.savefig('lake_superior_legend.png', bbox_inches='tight')
plt.close()  # Close the plot to avoid displaying it
