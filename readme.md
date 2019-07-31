HubFinder is a motif enumeration algorithm.
This repository provides python codes and experimental jupyter notebooks.

Genta Yoshimura, Atsunori Kanemura, Hideki Asoh,
[Enumerating Hub Motifs in Time Series Based on the Matrix Profile](https://milets19.github.io/papers/milets19_paper_5.pdf),
5th Workshop on Mining and Learning from Time Series (MiLeTS’19).

Oral presentation slide used in MiLeTS'19 is available at [slideshare]().

Please reference as
```
@misc{yoshimura19enumerating,
  title={Enumerating Hub Motifs in Time Series Based on the Matrix Profile},
  author={Genta Yoshimura, Atsunori Kanemura, Hideki Asoh},
  year={2019},
  publisher={Workshop on Mining and Learning from Time Series (MiLeTS)}
}
```

# Directory Structure
* `src`: Python codes and jupyter notebooks
    * `motif_synthetic.ipynb`: Experiment on synthesized time series
    * `motif_synthetic_complexity.ipynb`: Comparison of time complexities on synthesized time series
    * `motif_ecg.ipynb`: Experiment on ECG time series
    * `motif_motion.ipynb`: Experiment on human motion time series
    * `metric.py`: Time series metrics and matrix profile (STAMP/STOMP)
    * `motif.py`: Motif enumeration methods including HubFinder, SetFinder, and ScanMK
* `mitdb`: Dataset of ECG time series
* `motion-sense`: Dataset of human motion time series
* `figures`: Figures are saved this directory

# Requirement
* Python 3.5+
* numpy 1.13+
* pandas 0.20+
* matplotlib 2.0+
* scikit-learn 0.19+
* tqdm 4.15+
* wfdb 2.0+

# Usage
1. Download `106.atr`, `106.dat`, and `106.hea` from [PhisioNet](https://physionet.org/physiobank/database/mitdb/) and place them at `./mitdb/`.
2. Download `data_subjects_info.csv` and `B_Accelerometer_data.zip` from [MotionSense] and place thme at `./motion-sense/data/`.
3. Unzip `./motion-sense/data/B_Accelerometer_data.zip`.
4. Run each jupyter notebook `./src/*.ipynb`.

# License
Apache License 2.0
