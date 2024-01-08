import pandas as pd, numpy as np
import geopandas as gpd, rasterio as rio
from rasterio.mask import mask


csr_lat_long = "WGS84"
csr_moll = "+proj=moll"
# mask = 12
# src = 12


def save_data(
    data: pd.DataFrame, name: str, head_name: str = "shape_", path_dir="./ouptut/"
):
    name_file = head_name + name
    csv_file = name_file + ".csv"
    xls_file = name_file + ".xlsx"

    data.to_csv(path_dir + csv_file, index=False)
    # data.to_excel(path_dir + xls_file)


# epsg: 4326
def getFeatures(gdf: gpd.GeoDataFrame):
    """
    The function `getFeatures` takes a GeoDataFrame and returns a list of features in a format that is
    compatible with rasterio.

    :param gdf: A GeoDataFrame object that contains spatial data
    :return: a list containing the parsed features from the GeoDataFrame.
    """
    import json

    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def crop_raster(raster_path: str, shapefile: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    The function `crop_raster` takes a raster file path and a shapefile as inputs, crops the raster
    based on the shapefile, and returns a pandas DataFrame containing the x, y, and z values of the
    cropped raster.

    :param raster_path: The path to the raster file that you want to crop
    :type raster_path: str
    :param shapefile: The `shapefile` parameter is a GeoDataFrame object that represents the shapefile
    containing the geometries used for cropping the raster. It contains the spatial information and
    attributes associated with each geometry
    :type shapefile: gpd.GeoDataFrame
    :return: a pandas DataFrame containing the x, y, and z values of the cropped raster.
    """
    with rio.open(raster_path) as src:
        out_image, out_transform = mask(src, getFeatures(shapefile), crop=True)
    values = out_image.flatten()
    rows, cols = np.indices(out_image.shape[-2:])
    x, y = rio.transform.xy(out_transform, rows.flatten(), cols.flatten())

    # Crear un DataFrame con los valores y las coordenadas
    data = {"x": x, "y": y, "z": values}
    df = pd.DataFrame(data)
    return df


def points_inside(df: pd.DataFrame, shapefile) -> gpd.GeoDataFrame:
    points_df = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x"], df["y"])
    ).set_crs(epsg=4326)
    points_inside = gpd.sjoin(points_df, shapefile, how="inner", op="within")
    return points_inside


def simple_metrics(
    df: gpd.GeoDataFrame,
    target_name: str,
    ref: str = "z",
    cols: list[str] = ["index", "id_distr_b", "year", "newid", "baseline_P"],
    new_columns={"id_distr_b": "id_distr_bank", "baseline_P": "baseline_PSU"},
    not_null=0,
) -> pd.DataFrame:
    df = df.dropna(subset=[ref])
    ref_df = df[cols].iloc[:1]
    stats = df[ref].agg([np.mean, np.std, np.sum]).values.flatten()
    (
        ref_df[f"{target_name}_mean"],
        ref_df[f"{target_name}_sd"],
        ref_df[f"{target_name}_sum"],
    ) = stats
    ref_df = ref_df.replace([np.inf, -np.inf], not_null)
    return ref_df.rename(columns=new_columns)


def get_metrics(
    raster_path: str,
    data_shp: gpd.GeoDataFrame,
    target_name: str,
    ref: str = "z",
    cols: list[str] = ["index", "id_distr_b", "year", "newid", "baseline_P"],
    new_columns={"id_distr_b": "id_distr_bank", "baseline_P": "baseline_PSU"},
) -> pd.DataFrame:
    collect_data = pd.DataFrame()
    for i in range(len(data_shp)):
        row = data_shp.iloc[i : i + 1]
        df = crop_raster(raster_path, row)
        df = points_inside(df, row)
        stats = simple_metrics(
            df, target_name=target_name, ref=ref, cols=cols, new_columns=new_columns
        )
        collect_data = pd.concat((collect_data, stats))
    return collect_data
