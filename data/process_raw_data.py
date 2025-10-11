import pandas as pd
import numpy as np


def extract_data(file_path: str):
    with open(file_path, "r") as raw_data:
        data = []

        for line in raw_data.readlines():
            data.append(list(map(float, line.strip().split(" "))))

        return data


labels = [
    "mass",
    "pT_1",
    "pT_2",
    "px_1",
    "py_1",
    "pz_1",
    "energy_1",
    "px_2",
    "py_2",
    "pz_2",
    "energy_2",
]

signal = extract_data("signal (raw)/zpjj_signal-8D-mmjj400.0-mzp1000.0.txt")
csv_signal_data = pd.DataFrame(signal, columns=labels)
csv_signal_data.write_csv("signal.csv")


background = extract_data("background (raw)/ppuu_background-8D-mmjj400.txt")
csv_background_data = pd.DataFrame(background, columns=labels)
csv_background_data.write_csv("background.csv")
