import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')
sns.set()
sns.barplot(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y=[
            2714, 551, 76, 11, 1, 0, 0, 0, 0, 0])
plt.xlabel("Number of cells per droplet")
plt.ylabel("Number of droplets")
# plt.show()
plt.savefig("Figs/final_plot.png")
