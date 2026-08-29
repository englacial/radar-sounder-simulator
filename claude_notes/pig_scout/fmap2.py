import numpy as np, pandas as pd, xopr
c=xopr.OPRConnection(cache_dir="/home/thomasteisberg/Documents/coherent-radar-simulator/outputs/cache/xopr")
SEGS=[("2012_Antarctica_DC8","20121023_04"),("2009_Antarctica_DC8","20091020_06"),
      ("2009_Antarctica_DC8","20091020_03"),("2014_Antarctica_DC8","20141029_05"),
      ("2016_Antarctica_DC8","20161104_05"),("2018_Antarctica_DC8","20181107_01"),
      ("2011_Antarctica_DC8","20111014_07"),("2011_Antarctica_DC8","20111026_03")]
out=[]
for col,sp in SEGS:
    d=c.query_frames(collections=[col], segment_paths=[sp], exclude_geometry=True)
    d=pd.DataFrame(d)
    d['frame']=d['properties'].apply(lambda p:int(p['opr:frame']))
    d['t0']=pd.to_datetime(d['properties'].apply(lambda p:p['datetime']), format='mixed', utc=True).astype('int64')/1e9
    d['collection']=col; d['sp']=sp
    d['fid']=sp+'_'+d['frame'].map('{:03d}'.format)
    out.append(d[['collection','sp','frame','fid','t0']])
    print(col,sp,len(d),d.frame.min(),d.frame.max())
F=pd.concat(out).sort_values(['collection','sp','frame'])
F.to_csv("frame_times.csv",index=False)
print(F.head())
