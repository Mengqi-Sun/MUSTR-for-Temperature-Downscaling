# MUSTR

Daily urban air and surface temperature estimation at approximately 100 m resolution.
**Daily MUPNet** is the original name retained in the code; **MUSTR** is the final model name used in the paper.

## Repository

| Path | Contents |
| --- | --- |
| `MUSTR_Train/` | Model, data reader, training, losses, evaluation, and experiment assets. |
| `DataCube/schema.py` | Shared daily HDF5 format and validation. |
| `DataCube/samples/` | Twelve complete example cubes, station splits, and checksums. |
| `requirements.txt` | Tested Python dependencies. |

The examples cover 2018-2023 (2,191 days, UTC+8), with eight training and four validation locations from four regions. They include station air/GST labels, Landsat observations, and virtual points. Splits and data values are unchanged. This is a functional subset, not the full dataset used for the paper's results.

## Paper experiments

Configurations and run folders are under `MUSTR_Train/configs/abl_final/` and `MUSTR_Train/experiments/abl_final/`, respectively.

| Paper | Code name | Supervision | Metric source |
| --- | --- | --- | --- |
| E1 | `01_S_station_only` | Station air and surface temperature | E01 |
| E2 | `02_B_landsat_structure` | E1 + Landsat structure | E02 |
| E3 | `14_SP_station_physics` | E1 + physics | E14 |
| E4 (proposed) | `05_E_physics_no_absolute` | E1 + Landsat structure + physics | E05 |
| E5 | `10_E_physics0.05` | E4 with lower physics weight | E10 |
| E6 | `11_E_physics0.5` | E4 with higher physics weight | E11 |

Metric source `E05` means paper E4. Predictions are in Kelvin with `[minimum, mean, maximum]` channel order.
