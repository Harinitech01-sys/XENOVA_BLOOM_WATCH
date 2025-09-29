from data_acquisition import fetch_satellite_data
regions = ["India", "USA"]
for region in regions:
    print(fetch_satellite_data(region))
