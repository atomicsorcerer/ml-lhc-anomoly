import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 12})
plt.rcParams.update({"lines.linewidth": 1.5})
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# plt.rcParams.update({"font.serif": "Computer Modern Serif"})

signal = pd.read_csv("data/signal.csv")["mass"].to_numpy()
background = pd.read_csv("data/background.csv")["mass"].to_numpy()

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
