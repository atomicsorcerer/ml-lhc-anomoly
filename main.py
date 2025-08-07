import polars as pl
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 12})
plt.rcParams.update({"lines.linewidth": 1.5})
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# plt.rcParams.update({"font.serif": "Computer Modern Serif"})

signal = pl.read_csv("data/signal.csv").get_column("mass").to_numpy()
background = pl.read_csv("data/background.csv").get_column("mass").to_numpy()

plt.hist(
    [background, signal],
    bins=500,
    histtype="barstacked",
    color=["tab:blue", "tab:red"],
    label=["Background", "Signal"],
)
plt.xlabel("Mass")
plt.ylabel("Entries")
plt.legend()
plt.show()

plt.hist(
    [signal],
    bins=500,
    color="tab:red",
    label="Signal",
)
plt.xlabel("Mass")
plt.ylabel("Entries")
plt.legend()
plt.show()
