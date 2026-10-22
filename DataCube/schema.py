"""Ordered daily HDF5 contract, shared by the producer and training reader.

This module deliberately has no geospatial or machine-learning dependencies.
"""

SCHEMA_VERSION = "daily_spatial_v1"
TIME_ORIGIN = "2018-01-01"
TIMEZONE = "Asia/Shanghai"
INPUT_SIZE = 19
OUTPUT_SIZE = 5
STATIC = ("dem", "cmtbh", "svf", "kg")
TEXTURES = ("albedo", "ndvi", "ndbi", "sar_vv", "sar_vh", "ntl", "optical_age", "sar_age")
ERA5 = (
    "t2m_min", "t2m_mean", "t2m_max", "skt_min", "skt_mean", "skt_max",
    "d2m_mean", "u10_mean", "v10_mean", "ws10_mean", "ws10_day_mean",
    "ws10_prev_night_mean", "sp_mean", "tp_sum", "api_mean", "ssrd_mean",
    "ssrd_day_mean", "strd_mean", "strd_day_mean", "strd_prev_night_mean",
    "snowc_mean", "snowc_max", "snow_albedo_mean",
)
DERIVED_SPATIAL = ("q_air_mean", "rho_air_mean")
HOURS = tuple(f"era5_{var}_{stat}_hour" for var in ("t2m", "skt") for stat in ("min", "max"))
SOLAR = ("sunrise_hour", "sunset_hour", "daylight_duration_hours", "prev_night_duration_hours")
MET = ("tem_min", "tem_mean", "tem_max", "gst_min", "gst_mean", "gst_max")


def center_slice(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
    if any(n <= 0 or n % 2 != 1 for n in (input_size, output_size)) or input_size < output_size:
        raise ValueError("Input/output sizes must be positive odd numbers with input >= output")
    start = (input_size - output_size) // 2
    return slice(start, start + output_size)


def _era_unit(name):
    if name.startswith(("t2m_", "skt_", "d2m_")):
        return "K"
    if name.startswith(("u10_", "v10_", "ws10_")):
        return "m/s"
    if name.startswith(("ssrd_", "strd_")):
        return "W/m2"
    return "Pa" if name == "sp_mean" else "mm" if name in ("tp_sum", "api_mean") else "1"


# path -> (axis: static/annual/daily/vector/year, physical unit, stored dtype)
DATASETS = {
    **{f"static/{v}": ("static", "class_code" if v == "kg" else "1" if v == "svf" else "m",
                          "uint8" if v == "kg" else "float32") for v in STATIC},
    "annual/clcz": ("annual", "class_code", "uint8"),
    **{f"daily/{v}": ("daily", "day" if v.endswith("age") else "dB" if v.startswith("sar_")
                       else "unknown_native_radiance" if v == "ntl" else "1",
                       "int16" if v.endswith("age") else "float32") for v in TEXTURES},
    **{f"daily/era5/{v}": ("daily", _era_unit(v), "float32") for v in ERA5},
    "daily/derived/q_air_mean": ("daily", "kg/kg", "float32"),
    "daily/derived/rho_air_mean": ("daily", "kg/m3", "float32"),
    **{f"daily/derived/{v}": ("vector", "hour", "uint8") for v in HOURS},
    **{f"daily/derived/{v}": ("vector", "hour", "float32") for v in SOLAR},
    **{f"target/met/daily/{v}": ("vector", "Celsius", "float32") for v in MET},
    "target/lst_obs": ("daily", "K", "float32"),
    "target/lst_hour": ("vector", "hour", "float32"),
    "time/day_index": ("vector", f"days since {TIME_ORIGIN}", "int32"),
    "time/year": ("year", "year", "int16"),
    "coords/lon": ("static", "degrees_east", "float64"),
    "coords/lat": ("static", "degrees_north", "float64"),
}
ERA5_PATHS = tuple(f"daily/era5/{v}" for v in ERA5) + tuple(
    f"daily/derived/{v}" for v in DERIVED_SPATIAL + HOURS + SOLAR
)


def shape_for(axis, days, years, size=INPUT_SIZE):
    return {"static": (size, size), "annual": (years, size, size),
            "daily": (days, size, size), "vector": (days,), "year": (years,)}[axis]


def validate_header(h5):
    """Fail clearly on legacy cubes; never silently broadcast their ERA5 center."""
    if h5.attrs.get("Schema_Version") != SCHEMA_VERSION:
        raise ValueError(f"{h5.filename}: requires {SCHEMA_VERSION}; supply a compatible daily DataCube")
    if h5.attrs.get("Timezone") != TIMEZONE or h5.attrs.get("Time_Origin") != TIME_ORIGIN:
        raise ValueError(f"{h5.filename}: incorrect daily calendar metadata")
    days, years = len(h5["time/day_index"]), len(h5["time/year"])
    for path, (axis, unit, dtype) in DATASETS.items():
        if path not in h5:
            raise ValueError(f"{h5.filename}: missing {path}")
        ds = h5[path]
        if ds.shape != shape_for(axis, days, years) or ds.dtype.name != dtype:
            raise ValueError(f"{path}: invalid shape/dtype {ds.shape}/{ds.dtype}")
        if ds.attrs.get("unit") != unit:
            raise ValueError(f"{path}: expected unit {unit}, got {ds.attrs.get('unit')}")
    if "hourly" in h5:
        raise ValueError("Daily cubes must not contain hourly datasets")
