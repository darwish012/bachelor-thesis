# Machine Learning Based Analysis of Sports Movement

Bachelor thesis on collecting motion data with an ESP32-CAM and MPU6050 wearable sensor, then using a decision tree to distinguish human activities. The repository includes Arduino sketches for SD-card and Wi-Fi recording, collected accelerometer and gyroscope data, Python analysis scripts, and the submitted thesis. Tennis strokes are recorded and visualized; the classifier evaluates sitting, standing, walking, and stair movement.

## What is implemented

- Touch-controlled recording of timestamps and six motion channels.
- Offline CSV logging to an SD card, or TCP transmission to a Python CSV receiver.
- Raw signal plots and smoothing with a 20-sample moving average.
- Trial-level statistical feature extraction, standardization, and decision-tree classification across five randomized iterations.
- PCA plots, feature importance plots, activity accuracy charts, a confusion matrix, and a decision-tree plot.

## Repository structure

| Path | Contents |
| --- | --- |
| [5205930 SUBMITTED.pdf](5205930%20SUBMITTED.pdf) | Submitted thesis: hardware connections, PCB development, methods, figures, and results. |
| [datacollectingcodes/offlinemode/](datacollectingcodes/offlinemode/) | Arduino sketch that writes `/data.csv` to the SD card. |
| [datacollectingcodes/onlinemode/](datacollectingcodes/onlinemode/) | Wi-Fi sketch and `wifiesp32cam.py` TCP receiver. |
| [datasetcollected/](datasetcollected/) | Recorded CSV files for human activities and tennis strokes. |
| [datavisualization.py](codes%20for%20data%20visualization%20and%20anaylsis/datavisualization.py) | Raw and smoothed accelerometer/gyroscope plots. |
| [treeswithdws.py](codes%20for%20data%20visualization%20and%20anaylsis/treeswithdws.py) | Five-activity decision-tree analysis and evaluation plots. |

The CSV columns are `Time`, `Accel X`, `Accel Y`, `Accel Z`, `Gyro X`, `Gyro Y`, and `Gyro Z`. File prefixes mean sitting (`sit`), standing (`std`), walking (`wlk`), upstairs (`ups`), and downstairs (`dws`). Tennis recordings are `backhand.csv`, `forehand2.csv`, and `serve2.csv`.

## Run the analysis

Requirements: Python with pandas, NumPy, Matplotlib, seaborn, and scikit-learn. There is no dependency lockfile or specified Python version.

From the repository root:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
cd datasetcollected
python "../codes for data visualization and anaylsis/datavisualization.py"
python "../codes for data visualization and anaylsis/treeswithdws.py"
```

Run from `datasetcollected` because the scripts open CSV filenames relative to the working directory. Plots open interactively; the scripts do not save figure files. Classification prints per-iteration, per-activity, and overall trial-level accuracy. Random sampling has no fixed seed, so a new run can differ from the thesis results.

## Collect new data

Use Arduino IDE with ESP32 board support and an `MPU6050.h` library compatible with `initialize()` and `getMotion6()`. The sketches also use the ESP32 Wire, WiFi, and SD_MMC libraries. Hardware includes an ESP32-CAM, MPU6050, and touch sensor; offline recording requires an SD card.

The actual sketch definitions use SDA GPIO **16**, SCL GPIO **13**, and touch GPIO **12**. Some comments and thesis wiring text refer to SDA GPIO 14, so reconcile the wiring with the sketch before uploading. Open the chosen `.ino` sketch in Arduino IDE; the online sketch may need a matching sketch-folder name when opened.

For Wi-Fi recording, configure network credentials and the receiving computer's IP address in `onlinemodeesp32cam.ino`. In `wifiesp32cam.py`, change `CSV_FILE_PATH` from its original Windows desktop path to a writable destination. Start the receiver from the repository root:

```bash
python datacollectingcodes/onlinemode/wifiesp32cam.py
```

The receiver listens on port **12345**, matching the sketch. Use the touch sensor to start recording, stop recording, and reset the device. Offline and online sketches have loop delays of 10 ms and 50 ms respectively; these delays alone do not establish the actual sampling rate. The receiver writes different column names from the offline CSVs, so align its headers with the analysis scripts before analyzing a new online recording.

## Results

The [thesis](5205930%20SUBMITTED.pdf) reports an average overall activity-recognition accuracy of **95.79%** across five trials in Table 4.1. It also contains tennis and human-activity signal plots, PCA visualizations, and a confusion matrix. This is a reported experimental result, not a guarantee for a new run or new recordings.

## Tech stack

Arduino C++, ESP32-CAM, MPU6050, SD_MMC, Wi-Fi/TCP, Python, pandas, NumPy, scikit-learn, Matplotlib, and seaborn.

## Author

Abdelrahman Darwish ([GitHub @darwish012](https://github.com/darwish012)). The thesis names Prof. Dr. Mohamed Ehsan Ashour as supervisor and was submitted at the German University in Cairo on 19 May 2024.
