import os
import numpy as np
import rasterio as rio
import rasterio.mask as mask

from shapely.geometry import mapping
import matplotlib.pyplot as plt
import geopandas as gpd
import earthpy.spatial as es
import earthpy.plot as ep

raster_file = "./data/8_night_light/pk_night_light_harm.tiff"

with rio.open(raster_file) as src:
    lidar_chm = src.read(masked=True)[0]
    # extent = rio.plot.plotting_extend(src)
    # sop

ep.plot_bands(lidar_chm)
