import pandas as pd
import geopandas as gpd


csr_lat_long = "WGS84"
csr_moll = "+proj=moll"
mask = 12
src = 12


def save_data(
    data: pd.DataFrame, name: str, head_name: str = "shape_", path_dir="./ouptut/"
):
    name_file = head_name + name
    csv_file = name_file + ".csv"
    xls_file = name_file + ".xlsx"

    data.to_csv(path_dir + csv_file)
    data.to_excel(path_dir + xls_file)


def crop_raster_shapefile(raster_file, shp_file):
    shp_gpf = gpd.read_file(shp_file)
    out_image, out_transform = mask(src, shp_gpf.geometry, crop=True)
    raster_values = src.datasets_mask(out_image)
    values = [x for x in raster_values.flatten() if x != src.nodata]
    tbl_info = pd.DataFrame({"z": values})

    return tbl_info
