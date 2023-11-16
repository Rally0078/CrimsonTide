import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calendar import monthrange, month_name
from scipy.stats import describe, norm
import sys
import argparse


def getData(filename : str = "data.csv", verbose : bool = False) -> np.ndarray:
    df = pd.read_csv(filename)

    periodMonths = df.month
    periodDates = df.date  
    days = np.cumsum([monthrange(2023, i)[1] for i in range(1,13)])
    daysZeroed = days - days[0]

    periodDays = []

    for month, day in zip(periodMonths, periodDates):
        daysElapsed = daysZeroed[month - 1] + day
        periodDays.append(daysElapsed)
    if(verbose):
        print(periodDays)
    periodDays = np.array(periodDays)
    return periodDays

def plotPeriods(periodDays : np.ndarray, 
                showPlot : bool = False, 
                savePlot : bool = False):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    days = np.cumsum([monthrange(2023, i)[1] for i in range(1,13)])

    data_ar = np.zeros(shape=np.cumsum([monthrange(2023, i)[1] for i in range(1,13)])[-1])
    data_ar[periodDays] = 1
    ax.plot(periodDays, data_ar[periodDays], "ro", markersize=12)
    ax.plot(np.arange(0,365),data_ar, "b-")

    ax.set_title("Periods Data 2023")
    ax.set_xlabel("Day of the year")
    ax.set_ylabel("Periods (Yes/No)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])

    ax.set_xticks(days)
    ax.set_xticklabels(month_name[1:], fontsize=9)

    ax.margins(x=0.007, tight=True)
    ax.grid()
    if(savePlot):
        plt.savefig("periods2023.jpg", bbox_inches="tight", dpi=150)
    if(showPlot):
        plt.show()
    plt.close()

def findPeriodicity(periodDays : np.ndarray, 
                    start : int = 0, 
                    showPlot : bool = False, 
                    verbose : bool = False, 
                    savePlot : bool = False) -> np.ndarray:
    data_ar = np.zeros(shape=np.cumsum([monthrange(2023, i)[1] for i in range(1,13)])[-1])
    data_ar[periodDays] = 1
    acf = np.correlate(data_ar[start:], data_ar[start:], 'full')[-len(data_ar)+start:]
    inflection = np.diff(np.sign(np.diff(acf))) # Find the second-order differences
    peaks = (inflection < 0).nonzero()[0] + 1 # Find where they are negative
    delay = peaks[acf[peaks].argmax()] # Of those, find the index with the maximum value

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)

    ax.plot(acf[:60], "ro-")

    ax.set_title(f"Autocorrelation Function (Approximate period = {delay} days)")
    ax.set_xlabel("Days")

    ax.set_xticks(np.concatenate((np.arange(0,35,2), np.arange(35,70, 5))))
    ax.grid()
    if(savePlot):
        plt.savefig("acfperiods2023.jpg", dpi=150, bbox_inches="tight")
    if(showPlot):
        plt.show()
    plt.close()
    if(verbose):
        print(f"Periodicity of periods by autocorrelation = {delay} days")
        print(f"First order time periods are : {peaks[0:4]}")
    return peaks[0:4]

def plotHistogram(periodDays : np.ndarray, 
                  showPlot : bool = False, 
                  savePlot : bool = False,
                  verbose : bool = False):
    diffDays = periodDays[1:]-periodDays[:-1]

    _, _, mean, var, _, _ = describe(diffDays)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    if(verbose):
        print(f"Time period between menstruation = {diffDays}")
    ax.hist(diffDays, bins=np.arange(25, 33), width=0.5, align="mid", density=True)
    ax.set_title(f"Distribution of menstruation periods ($\mu = $ {mean} days, $\sigma = $ {np.sqrt(var):.3f} days)")
    x = np.linspace(24, 35, 1000)
    ax.plot(x, norm.pdf(x, mean, np.sqrt(var)), linestyle="dashed", color="red")
    ax.vlines(mean - np.sqrt(var), 0, 0.35, linestyle="dashed", color="green", label=f"$1-\sigma$ from mean")
    ax.vlines(mean + np.sqrt(var), 0, 0.35, linestyle="dashed", color="green")
    ax.set_xlabel("Number Days between menstruation")
    ax.set_ylabel("Relative frequency of occurence")
    ax.margins(x=0)
    ax.set_xticks(np.arange(25,33))
    ax.set_xlim([24,33])
    ax.legend()
    ax.grid()
    if(savePlot):
        plt.savefig("periods2023hist.jpg", dpi=150, bbox_inches="tight")
    if(showPlot):
        plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Menstrual Cycle Tracker",
        description="Tracks menstrual cycle and estimates the periodicity. Looks for data by default in 'data.csv'"
    )
    parser.add_argument("-f", "--file", type=str, action="store", help="The name of the csv file containing period data", required=False)
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help = "Shows console information")
    parser.add_argument("-s", "--showplots", action="store_true", required=False, help = "Shows plots in a window")
    parser.add_argument("-o", "--out", action="store_true", required=False, help = "Saves plots to .jpg files")
    args = parser.parse_args()
    file = "data.csv"
    if(args.file == None):
        file = "data.csv"
    else:
        file = args.file
    if(args.verbose):
        print(f"Arguments: {args.file, args.verbose, args.showplots, args.out}")
    periodDays = getData(filename = file, verbose = args.verbose)
    plotPeriods(periodDays = periodDays, showPlot=args.showplots, savePlot = args.out)
    periods = findPeriodicity(periodDays=periodDays, showPlot = args.showplots, verbose = args.verbose, savePlot = args.out)
    if (args.showplots or args.out):
        plotHistogram(periodDays = periodDays, showPlot = args.showplots, verbose = args.verbose, savePlot = args.out)