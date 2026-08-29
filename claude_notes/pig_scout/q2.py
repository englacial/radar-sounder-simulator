import xopr, shapely.geometry as sg, pandas as pd, json
c = xopr.OPRConnection(cache_dir="/home/thomasteisberg/Documents/coherent-radar-simulator/outputs/cache/xopr")
cols = [x['id'] for x in c.get_collections() if 'Antarctica' in x['id']]
print(len(cols), sorted(cols))
# Pine Island box (generous): lon -108..-92, lat -77.5..-74.0
poly = sg.box(-108, -77.5, -92, -74.0)
df = c.query_frames(collections=cols, geometry=poly)
print(df.shape)
print(df.columns.tolist())
df.to_pickle("pig_frames.pkl")
print(df.groupby('collection').size().sort_values(ascending=False))
