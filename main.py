import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from dpca import DensityPeakCluster
from shared_nn import aSNNC
from sklearn.cluster import DBSCAN

# Import Dataset

df = pd.read_csv('data/Compound_data.csv', names = ['x1', 'x2'])


# Parameters
pct = float(input('Enter pct: '))
delta_t = float(input('Enter delta_t: '))


# Representative Selection and Initial Cluster Formation
dpca = DensityPeakCluster(percent = pct, density_threshold = 0, distance_threshold = delta_t, anormal = False)

dpca.fit(df.iloc[:, :])

# Plot Initial Cluster Formation
# sns.scatterplot(x = df.iloc[:, 0], y = df.iloc[:, 1], hue = dpca.labels_, palette = "bright")
# plt.show()


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

    # Perform aSNNC in l1
    asnnc = aSNNC(l1_datapoints)
    # print(asnnc.labels)
    labels = [tuple(label) for label in asnnc.labels]
    
    # Print aSNNC results of l1 datapoints
    # sns.scatterplot(x = l1_datapoints[:, 0], y = l1_datapoints[:, 1], hue = labels)
    # plt.show()

    # Final clusters will be stored in the form {centre: [datapoints]}
    final_clusters = {}

    # Iterate through each asnnc cluster to find boundary points
    bp_threshold = len(l1_datapoints) / len(asnnc.centers)
    
    boundary_points = []
    dpca_labels = dpca.labels_

    dpca_reps = dpca.center
    l1_reps = level_wise_reps[0][0, :]
    lp_reps = list(set(dpca_reps).difference(set(l1_reps)))

    
    for center in asnnc.centers:
        count = 0
        for label in labels:
            if tuple(label) == tuple(center):
                count += 1

        if count < bp_threshold:
            # Merge boundary points to closest cluster in lp
            cluster_points = []
            for i in range(len(l1_datapoints)):
                if tuple(labels[i]) == tuple(center):
                    closest_lp_rep = None
                    closest_lp_rep_dist = 1000000000
                    for rep in lp_reps:
                        distance = np.linalg.norm(np.array(datapoints[rep]) - np.array(l1_datapoints[i]))
                        if distance < closest_lp_rep_dist:
                            closest_lp_rep_dist = distance
                            closest_lp_rep = rep
                        
                    for j in range(datapoints.shape[0]):
                        if tuple(datapoints[j]) == tuple(l1_datapoints[i]):
                            dpca_labels[i] = closest_lp_rep
        else:
            cluster_points = []
            for i in range(len(l1_datapoints)):
                if tuple(labels[i]) == tuple(center):
                    cluster_points.append(l1_datapoints[i])

            final_clusters[tuple(center)] = cluster_points


    # For higher density levels > 1
    for density_level in range(1, numl):
        print('\nDensity Level:', density_level)
        # Consolidate all datapoints in this density level.
        current_reps = level_wise_reps[density_level]
        current_density_dps = []
        for i in range(datapoints.shape[0]):
            if dpca_labels[i] in current_reps[0]:
                current_density_dps.append(datapoints[i])

        # Separate out cluster with lowest local density in this level
        C_low = []
        x_low = current_reps[0, -1]
        x_far = None
        max_dist = 0
        for i in range(datapoints.shape[0]):
            if dpca_labels[i] == x_low:
                C_low.append(datapoints[i])
                distance = np.linalg.norm(datapoints[i] - datapoints[int(x_low)])
                if distance > max_dist:
                    max_dist = distance
                    x_far = datapoints[i]

        sim = np.array([np.linalg.norm(x_far - xi) for xi in current_density_dps])
        sim = np.sort(sim)
        
        Eps = sim[math.ceil(len(C_low))]
        print('Eps:', Eps)

        # Separate out cluster with highest local density in this level
        C_high = []
        x_high = current_reps[0, 0]
        for i in range(datapoints.shape[0]):
            if dpca_labels[i] == x_high:
                C_high.append(datapoints[i])

        temp_distances = np.array([np.linalg.norm(datapoints[int(x_high)] - xj) - Eps for xj in C_high])
        MinPts_high = (temp_distances >= 0).sum()

        temp_distances = np.array([np.linalg.norm(x_far - xj) - Eps for xj in C_low])
        MinPts_low = (temp_distances >= 0).sum()
        
        MinPts = math.ceil((MinPts_high + MinPts_low) / 2)
        print('MinPts:', MinPts)

        X = np.stack(current_density_dps)
        dbscan = DBSCAN(eps = 2, min_samples = MinPts)
        labels = dbscan.fit_predict(X)

        # sns.scatterplot(x = X[:, 0], y = X[:, 1], c=labels, s=50)
        # plt.show()
        # print(labels)