from collections import defaultdict
import numpy as np
import math

class DisjointSet:
    def __init__(self, datapoints):
        self.parent = defaultdict(lambda: None)
        self.rank = defaultdict(lambda: 0)

        for dp in datapoints:
            self.parent[tuple(dp)] = None

    def find(self, x):
        x = tuple(x)
        if self.parent[x] is None:
            return x
        self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        x = tuple(x)
        y = tuple(y)
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Union by rank for balancing
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1



class aSNNC:
    def __init__(self, datapoints):
        self.datapoints = datapoints
        self.k = int(math.sqrt(datapoints.shape[0]))

        self.centers = []
        self.labels = []

        self.calc_snnc()


    def knn(self, X):
        nearest_neighbors = []

        distances = np.linalg.norm(X - self.datapoints, axis = 1)
        nearest_indices = np.argsort(distances)[1:self.k+1]

        k_neighbours = self.datapoints[nearest_indices]

        return np.array(k_neighbours)


    def calc_snnc(self):
        d_set = DisjointSet(self.datapoints)
        n = self.datapoints.shape[0]
        for i in range(0, n-1):
            for j in range(i+1, n):
                point1 = self.datapoints[i]
                point2 = self.datapoints[j]
                if d_set.find(point1) != d_set.find(point2):
                    kn1 = self.knn(point1)
                    kn2 = self.knn(point2)

                    set1 = set(tuple(neighbour) for neighbour in kn1)
                    set2 = set(tuple(neighbour) for neighbour in kn2)

                    common = len(set1.intersection(set2))
                    if common > 1:
                        d_set.union(point1, point2)

        for point in self.datapoints:
            parent = np.array(d_set.find(point))
            self.labels.append(parent)

            if not np.isin(parent, self.centers).all():
                self.centers.append(parent)

        # Convert parent to labels