import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dpca import DensityPeakCluster


# Import Dataset

df = pd.read_csv('data/Compound_data.csv', names = ['x1', 'x2'])


# Parameters
pct = float(input('Enter pct: '))
delta_t = float(input('Enter delta_t: '))


# Representative Selection and Initial Cluster Formation
dpca = DensityPeakCluster(percent = pct, density_threshold = 0, distance_threshold = delta_t, anormal = False)

dpca.fit(df.iloc[:, :])

# Plot Initial Cluster Formation
sns.scatterplot(x = df.iloc[:, 0], y = df.iloc[:, 1], hue = dpca.labels_, palette = "bright")
plt.show()


# Dividing Representatives into Density Levels
rho_list = dpca.rho

total_levels = 10

rep_densities = rho_list[dpca.center]

indices = np.argsort(rep_densities)
centers = np.array(dpca.center)[indices]

rep_densities = rep_densities[indices]
representatives = np.array([centers, rep_densities])

max_rho = max(representatives[1])
min_rho = min(representatives[1])
range_width = (max_rho - min_rho) / total_levels

print('Width:', range_width)

# Plot Density Levels Chart
# sns.scatterplot(x = representatives[1], y = np.zeros(representatives[1].shape[0]))
# plt.show()

n = df.shape[0]
level_wise_reps = []
current_level_reps = []
current_level_reps.append([representatives[0, 0], representatives[1, 0]])
for i in range(1, representatives.shape[1]):
    if representatives[1, i] - representatives[1, i-1] >= 2*range_width:
        current_level_reps = np.array(current_level_reps)
        level_wise_reps.append(current_level_reps.T)
        current_level_reps = []
    else:
        current_level_reps.append([representatives[0, i], representatives[1, i]])
current_level_reps = np.array(current_level_reps)
level_wise_reps.append(current_level_reps.T)

# for i in range(len(level_wise_reps)):
#     print('Level', i)
#     print(level_wise_reps[i])

numl = len(level_wise_reps)
print('Number of Levels: ', numl)


# Final Cluster Formation

if numl == 1:
    # Simply return the initially formed clusters
    sns.scatterplot(x = df.iloc[:, 0], y = df.iloc[:, 1], hue = dpca.labels_, palette = "bright")
    plt.show()

else:
    # Step 1: Finding Boundary Points in l1
    # Get all l1 datapoints together
    datapoints = df.values
    l1_datapoints = []
    for i in range(datapoints.shape[0]):
        datapoint = datapoints[i]
        label = dpca.labels_[i]

        if label in level_wise_reps[0][0]:
            l1_datapoints.append(datapoint)

    l1_datapoints = np.stack(l1_datapoints)
    # Plot Level 1 Datapoints
    # sns.scatterplot(x = l1_datapoints[:, 0], y = l1_datapoints[:, 1])
    # plt.show()

    