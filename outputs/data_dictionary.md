# Verified Data Dictionary — ETH Zurich Plant Stress Phenotyping Dataset

This dictionary defines all validated tables, columns, modalities, physical units, and valid domains.

### 1. `box-weights-kg.csv`
| Column Name | Type | Unit | Description | Valid Domain | Missingness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Box Number` | Integer | None | Experimental box identifier | 1 – 31 | 0% |
| `DD.MM.YYYY` (18 columns) | Float | Kilograms (kg) | Total weight of box (soil + plants + moisture) | 6.52 – 19.88 kg | 0% |

### 2. `greenhouse-temperature-humidity-logs.csv`
| Column Name | Type | Unit | Description | Valid Domain | Missingness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Date` | String | DD.MM.YY | Date of greenhouse logging | 19.01.18 – 10.04.18 | 0% |
| `Time` | String | HH:MM:SS | Time of greenhouse logging (12-min intervals) | 00:00:00 – 23:48:00 | 0% |
| `Room Temperature (degree Celsius)` | Float | °C | Ambient room temperature | 11.7 – 25.9 °C | 0% |
| `Relative Humidity (%)` | Float | % | Ambient relative humidity | 26.2 – 79.9 % | 0% |

### 3. `spad-measurements.csv`
| Column Name | Type | Unit | Description | Valid Domain | Missingness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Box Number` | Integer | None | Experimental box identifier | 1 – 31 | 0% |
| `DD.MM.YYYY` (5 dates) | Float | SPAD Unit | Soil-Plant Analyses Development index (chlorophyll proxy) | 29.1 – 59.7 | 12.9% (Boxes 16-18, 31 have no sugarbeet) |

### 4. `yield-weight-grams.csv`
| Column Name | Type | Unit | Description | Valid Domain | Missingness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Box Number` | Integer | None | Experimental box identifier | 1 – 31 | 0% |
| `Fresh leaves weight` | Float | Grams (g) | Above-ground canopy fresh biomass at harvest | 249.75 – 1662.40 g | 12.9% (Boxes 16-18, 31 have no sugarbeet) |
| `Dry leaves weight` | Float | Grams (g) | Above-ground canopy dry biomass at harvest | 35.40 – 203.40 g | 12.9% |
| `Fresh beet weight` | Float | Grams (g) | Below-ground beet root fresh weight at harvest | 83.28 – 558.01 g | 12.9% |
| `Dry beet weight` | Float | Grams (g) | Below-ground beet root dry weight at harvest | 11.65 – 94.64 g | 12.9% |

### 5. Image Modalities (`extracted/images/{date}/{modality}/{box}.png`)
| Modality Directory | Sensor Hardware | Resolution | Color Channels | Bit Depth | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rgb/` | Intel RealSense ZR300 RGB | 1920 x 1080 | 3 (RGB) | 8-bit | High-resolution top-down color image of box canopy |
| `ir1/` | Intel RealSense ZR300 Infrared | 640 x 480 | 1 (Grayscale) | 8-bit | Left infrared stereo camera view |
| `ir2/` | Intel RealSense ZR300 Infrared | 640 x 480 | 1 (Grayscale) | 8-bit | Right infrared stereo camera view |
| `multispec/` | Ximea 5x5 Snapshot Mosaic | 2048 x 1088 | 1 (Raw Mosaic) | 8-bit | 25 spectral bands (600–875 nm) in 5x5 repeating mosaic |
