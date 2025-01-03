import pandas as pd
import matplotlib.pyplot as plt
import math
import random

def plot_any_same_person(how_many, folder_name, conc_range = 1):
    for a in range(how_many):
        theguy = int(random.random()*2000)
        print("chosen guy is: ", theguy)
        for b in range(conc_range):
            df = pd.read_csv(folder_name + "/" + str(theguy + (2000*(b))) + "_pace.csv")
            plt.plot(df["Time"], df["Vm"], label = "ID " + str(theguy) + " cmax " + str(b+1))
    plt.legend()

plot_any_same_person(5, "./result/post_IC50_bepridil_simple_1_2", 2)
plt.savefig("plot.png")