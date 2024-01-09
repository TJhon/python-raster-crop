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


# General functions
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


def points_inside(df: pd.DataFrame, shapefile, not_null=10000000) -> gpd.GeoDataFrame:
    df = df[abs(df["z"]) < not_null]
    df = df[df["z"] > 0]
    points_df = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x"], df["y"])
    ).set_crs(epsg=4326)
    points_inside = gpd.sjoin(points_df, shapefile, how="inner", op="within")

    return points_inside


# epsg: 4326
def simple_metrics(
    df: gpd.GeoDataFrame,
    target_name: str,
    ref: str = "z",
    cols: list[str] = ["index", "id_distr_b", "year", "newid", "baseline_P"],
    new_columns={"id_distr_b": "id_distr_bank", "baseline_P": "baseline_PSU"},
) -> pd.DataFrame:
    # df = df.dropna(subset=[ref])
    ref_df = df[cols].iloc[:1]
    stats = df[ref].agg([np.mean, np.std, np.sum]).values.flatten()
    (
        ref_df[f"{target_name}_mean"],
        ref_df[f"{target_name}_sd"],
        ref_df[f"{target_name}_sum"],
    ) = stats
    # ref_df = ref_df.replace([np.inf, -np.inf], not_null)
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


# mollwide


def lat_to_moll(shp_file: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return shp_file.to_crs(epsg=9001)


def get_info_settlement(
    main_table: pd.DataFrame,
    prefix: str = "dummy_settl_",
    col_z="z",
    name_c="value",
    id_cols: list[str] = ["index", "id_distr_b", "year", "newid", "baseline_P"],
    row_info=None,
):
    left_df = main_table[id_cols].iloc[:1]
    if row_info is not None:
        left_df = row_info
    info = (
        main_table.groupby(col_z).size().div(len(main_table)).reset_index(name=name_c)
    )
    info["z"] = prefix + info["z"].astype(str)
    right_df = pd.pivot_table(
        info, values="value", columns="z", aggfunc=np.sum, fill_value=0
    )

    all = pd.concat([left_df, right_df], ignore_index=True)
    return all.ffill().bfill().head(1)


def get_metrics_setl(
    raster_path: str,
    data_shp: gpd.GeoDataFrame,
    prefix_name: str = ["dummy_settl_", "dummy_settle_without_na_"],
    col_z: str = "z",
    cols: list[str] = ["index", "id_distr_b", "year", "newid", "baseline_P"],
    new_columns={"id_distr_b": "id_distr_bank", "baseline_P": "baseline_PSU"},
) -> pd.DataFrame:
    with_nas = pd.DataFrame()
    without_nas = pd.DataFrame()
    data_shp = data_shp.to_crs(epsg=9001)
    for i in range(len(data_shp)):
        row = data_shp.iloc[i : i + 1]

        row_info = row[cols].head(1)

        df = crop_raster(raster_path, row)
        df = points_inside(df, row)
        with_na = get_info_settlement(
            df,
            prefix=prefix_name[0],
            col_z=col_z,
            name_c="value",
            id_cols=cols,
            row_info=row_info,
        )
        without_na = get_info_settlement(
            df.query("z > 0"),
            prefix=prefix_name[1],
            col_z=col_z,
            name_c="value",
            id_cols=cols,
            row_info=row_info,
        )

        with_nas = pd.concat((with_nas, with_na))
        without_nas = pd.concat((without_nas, without_na))
    with_nas = with_nas.fillna(value=0).rename(columns=new_columns)
    without_nas = without_nas.fillna(value=0).rename(columns=new_columns)
    return with_nas, without_nas
