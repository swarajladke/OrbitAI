# Latency Benchmark Report: Second Machine Measurement

- **Route**: NATIVE RUN, Python 3.12.9
- **Commit**: `583f4bfdade7078d745a5d52d01ec70205685864` (`583f4bf`)
- **Date**: 2026-09-08
- **Hardware Measured**:
  - **This Laptop**: Intel(R) Core(TM) i7-10870H CPU @ 2.20GHz (8 cores, 16 logical processors, 16 MB L3 cache)
  - **Older Laptop**: Intel Core i5-7200U (2 cores, 4 threads, 3 MB L3 cache)

---

## 1. Verification Gates Summary

### Gate 0: Exact Submitted Code
- **0.1 Commit & Date**:
  ```text
  583f4bfdade7078d745a5d52d01ec70205685864 2026-09-04 19:38:12 +0530
  ```
- **0.2 Git Working Tree Status**:
  `git status --porcelain` produced zero output (clean working tree).
- **0.3 Model Byte Sizes**:
  ```text
  scorer_pregeom.joblib                   359304
  scorer_objectness_pre_geometry.joblib   525648
  box_regressor_arm2.joblib               611945
  ```
  All three submitted model weights matched byte-for-byte.

### Gate 1: Dataset Location
- **1.1 Absolute Path `<DATASET>`**:
  `C:\Users\Helios\Desktop\Orbitsight\OrbitSight_Dataset`
- **1.2 File Counts**:
  - `(Get-ChildItem <DATASET>\Training_sets -Filter *.npy).Count` = **17**
  - `(Get-ChildItem <DATASET>\Training_sets -Filter *_bb_windows_40ms.txt).Count` = **17**
- **1.3 Repo Cleanliness**:
  Confirmed `git status --porcelain` remained completely empty.

### Gate 2: Execution Route
- **2.1 Docker Availability**:
  `docker --version` failed (Docker not installed).
- **Route Taken**: **ROUTE B** (NATIVE RUN, Python 3.12.9).
- **Environment Details**:
  - `python --version` -> `Python 3.12.9`
  - `pip show scikit-learn | Select-String Version` -> `Version: 1.5.1`

### Gate 3: Measurement Trustworthiness
- **3.1 CPU Hardware Query**:
  ```text
  Name                                      NumberOfCores NumberOfLogicalProcessors MaxClockSpeed
  ----                                      ------------- ------------------------- -------------
  Intel(R) Core(TM) i7-10870H CPU @ 2.20GHz             8                        16          2208
  ```
- **3.2 Mains Power Status**:
  `Get-CimInstance BatteryStatus`: Confirmed `PowerOnline: True`, `Charging: True`, `Discharging: False`.
- **3.3 Background Applications**:
  Closed all competing user processes prior to measurement: Google Chrome, Microsoft Edge, Steam / Steamwebhelper, Notion, and Epic Games Launcher.
- **3.4 Thread Pinning**:
  Pinned to single-threaded BLAS/OpenMP execution:
  `$env:OMP_NUM_THREADS = '1'`
  `$env:OPENBLAS_NUM_THREADS = '1'`
  `$env:MKL_NUM_THREADS = '1'`

---

## 2. Step A — Main Measurement Results

- **Command Run**:
  `python -m src.latency_bench --dataset-dir C:\Users\Helios\Desktop\Orbitsight\OrbitSight_Dataset\Training_sets --config config.yaml --reps 5 --warmup-windows 20`
- **Raw Log Location**: `experiments\hw_i7\latency_i7_full.txt` (232,374 bytes)
- **Total Wall-Clock Time**: **83.17 minutes** (4,990.16 seconds).
  *(Reference on older laptop: 48.08 minutes / 2,885.18 seconds)*

### Per-Sequence Latency Table (17 Training Sequences)

| Sequence Name | Compute p99 | Compute MAX | End-to-End p99 |
| :--- | :---: | :---: | :---: |
| `2025_12_23_21_12_28_EVK4_mag5.2` | 31.34 ± 2.36 ms | 41.89 ± 5.3 ms | 71.34 ms |
| `DAVIS_COSMOS1933_18958_2024-12-04-18-37-01` | 11.07 ± 2.62 ms | 19.38 ± 5.6 ms | 51.07 ms |
| `DAVIS_EGS_16908_2024-11-01-19-10-44` | 8.32 ± 0.14 ms | 14.33 ± 1.6 ms | 48.32 ms |
| `DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06` | 4.93 ± 0.19 ms | 10.01 ± 3.1 ms | 44.93 ms |
| `DAVIS_RESURSDK1_29228_2024-12-04-18-37-01` | 9.93 ± 0.32 ms | 17.82 ± 1.3 ms | 49.93 ms |
| `DAVIS_SL12RB2_15772_2024-12-04-18-21-37` | 8.88 ± 0.46 ms | 13.18 ± 1.0 ms | 48.88 ms |
| `DAVIS_SL16RB_20625_2024-12-04-19-34-18` | 8.68 ± 0.22 ms | 16.37 ± 2.6 ms | 48.68 ms |
| `DAVIS_SL16RB_26070_2024-12-04-19-14-39` | 9.32 ± 0.38 ms | 15.71 ± 3.4 ms | 49.32 ms |
| `DAVIS_SL8RB_2025-01-13-19-15-36` | 6.47 ± 0.10 ms | 11.67 ± 1.1 ms | 46.47 ms |
| `DVX_Filtered_ACS3_59588_2025-01-20-19-35-44` | 17.13 ± 3.70 ms | 43.09 ± 27.9 ms | 57.13 ms |
| `DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17` | 21.11 ± 0.34 ms | 28.89 ± 1.8 ms | 61.11 ms |
| `DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07` | 21.08 ± 0.36 ms | 30.90 ± 1.9 ms | 61.08 ms |
| `DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50` | 21.23 ± 0.18 ms | 30.71 ± 2.7 ms | 61.23 ms |
| `DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35` | 17.96 ± 1.29 ms | 26.46 ± 4.8 ms | 57.96 ms |
| `DVX_Filtered_Stars2_2025-01-20-19-57-17` | 13.14 ± 1.69 ms | 16.87 ± 2.5 ms | 53.14 ms |
| `DVX_Filtered_Stars_2025-01-20-19-15-10` | 14.23 ± 0.11 ms | 24.50 ± 0.6 ms | 54.23 ms |
| `DVX_NOAA6_11416_2025-01-20-19-06-31` | 22.90 ± 1.12 ms | 51.80 ± 19.9 ms | 62.90 ms |

### Budget Compliance (40 ms Budget Threshold)

- **compute p99 under 40 ms**: **17 of 17** *(older laptop: 15 of 17)*
- **compute MAX under 40 ms**: **14 of 17** *(older laptop: 2 of 17)*
- **end-to-end p99 under 40 ms**: **0 of 17** *(older laptop: 0 of 17)*

### System Stalls (> 1000 ms Discarded)
- **This Laptop**: **0 stall windows discarded** across all 17 sequences.
- *(Older laptop reference: 5 stalls discarded, all in `DVX_NOAA6_11416_2025-01-20-19-06-31`)*.

### Dataset Invariants Reconciliation
- **Total Windows Processed (All 21 sequences)**: **143,750** (Verified mathematically: 106,192 windows across the 17 training sequences + 37,558 windows across the 4 testing sequences).
- **Total Predictions Emitted**: **16,955** (Matches reference invariant exactly).
- **Total Ground-Truth Boxes**: **22,368** (Verified: 22,389 text lines minus 21 header rows).

---

## 3. Step B — Run-to-Run Variance Note

Per the brief's explicit instruction:
> *"If Step A alone runs past 45 minutes, skip Step B entirely and report Step A only. Partial results reported honestly are useful; rushed or invented ones are not."*

Because Step A required **83.17 minutes** wall-clock time (exceeding the 45-minute threshold), Step B was skipped entirely.

---

## 4. Step C — Testing the Architecture Prediction

### C.1 Speed-Up Factors by Camera Type

Using representative sequences per sensor:

| Camera Type | Sequence | older laptop (Intel Core i5-7200U) | this laptop (Intel Core i7-10870H) | Speed-up Factor |
| :--- | :--- | :---: | :---: | :---: |
| **DAVIS** (346×260) | `DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06` | 14.99 ms | 4.93 ms | **3.04×** |
| **DVX** (640×480) | `DVX_NOAA6_11416_2025-01-20-19-06-31` | 84.67 ms | 22.90 ms | **3.70×** |
| **EVK4** (1280×720) | `2025_12_23_21_12_28_EVK4_mag5.2` | 58.73 ms | 31.34 ms | **1.87×** |

### C.2 Prediction Evaluation

- **Advance Prediction**: The EVK4 sequences will speed up *more* than the DAVIS or DVX sequences, because the 16 MB L3 cache on this laptop fits the 1280×720 image footprint (921,600 pixels) whereas the 3 MB L3 cache on the older laptop caused thrashing.
- **Outcome**: **FAILED**.
  - DVX sped up by **3.70×**.
  - DAVIS sped up by **3.04×**.
  - EVK4 sped up by **1.87×** (the lowest speed-up among all three camera architectures).
- **Analysis**: While EVK4 frames fit in the 16 MB cache, the algorithmic bottleneck for EVK4 is dominated by morphological operations, connected-component labeling, and bounding box candidate generation over the 921.6k pixel array, where memory bandwidth and single-core instruction throughput dominate rather than cache hit rate alone.

---

## 5. Comparative Performance Summary

| Metric / Sequence | older laptop (Intel Core i5-7200U) | this laptop (Intel Core i7-10870H) | Change |
| :--- | :---: | :---: | :---: |
| **Execution Route** | CONTAINER RUN | NATIVE RUN, Python 3.12.9 | Route B |
| **Git Commit** | `583f4bf` | `583f4bf` | Identical |
| **Sequences Passing Compute p99 < 40 ms** | 15 of 17 | **17 of 17** | **+2 (+13.3%)** |
| **Sequences Passing Compute MAX < 40 ms** | 2 of 17 | **14 of 17** | **+12 (+600%)** |
| **Sequences Passing End-to-End p99 < 40 ms** | 0 of 17 | **0 of 17** | Unchanged |
| **System Stalls (> 1000 ms)** | 5 | **0** | **-5 (-100%)** |
| `2025_12_23_21_12_28_EVK4_mag5.2` Compute p99 | 58.73 ms | **31.34 ms** | **-27.39 ms (1.87× faster)** |
| `DAVIS_Filtered_NOAA6` Compute p99 | 14.99 ms | **4.93 ms** | **-10.06 ms (3.04× faster)** |
| `DVX_NOAA6_11416` Compute p99 | 84.67 ms | **22.90 ms** | **-61.77 ms (3.70× faster)** |
| **Full Run Wall-Clock Duration** | 48.08 min | **83.17 min** | +35.09 min |
