import geopandas as gpd

# Load taxi zones
zones = gpd.read_file('Data/zones/taxi_zones.shp')

print("=" * 60)
print("NYC TAXI ZONES COUNT")
print("=" * 60)
print(f"Total zones in shapefile: {len(zones)}")
print(f"Zone IDs range: {zones['LocationID'].min()} to {zones['LocationID'].max()}")
print(f"Unique zone IDs: {zones['LocationID'].nunique()}")
print("\nZone ID list:")
print(sorted(zones['LocationID'].tolist()))
