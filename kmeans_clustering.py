import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

star_data = pd.read_csv("star_data.csv")

mass_radius_column = star_data.iloc[:, 2:4].values

within_cluster_sum_of_squares = []
for k in range(1, 9):
    k_means = KMeans(n_clusters = k, random_state = 42)
    k_means.fit(mass_radius_column)
    within_cluster_sum_of_squares.append(k_means.inertia_)

plt.figure(figsize = (10, 5))
sns.lineplot(x = range(1, 9), y = within_cluster_sum_of_squares, markers = 'bx-')

plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of Squares')

k_means = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
prediction = k_means.fit_predict(mass_radius_column)

plt.figure(figsize = (10, 5))
sns.scatterplot(x = mass_radius_column[prediction == 0, 0], y = mass_radius_column[prediction == 0, 1], color = 'orange', label = 'Star Cluster 1')
sns.scatterplot(x = mass_radius_column[prediction == 1, 0], y = mass_radius_column[prediction == 1, 1], color = 'blue', label = 'Star Cluster 2')
sns.scatterplot(x = mass_radius_column[prediction == 2, 0], y = mass_radius_column[prediction == 2, 1], color = 'green', label = 'Star Cluster 3')
sns.scatterplot(x = k_means.cluster_centers_[:, 0], y = k_means.cluster_centers_[:, 1], color = 'red', label = 'Centroids', s = 100, marker = ',')

plt.title('Clusters of Stars')
plt.xlabel('Mass of Stars')
plt.ylabel('Radius of Stars')
plt.legend()

plt.show()