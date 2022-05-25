import numpy as np
from sklearn.cluster import KMeans
from fcmeans import FCM
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def run_fuzzy_c_means(num_cluster:int, data:np.ndarray, capacity:int, increase_vehicle=False):
    """
        num_cluster: number of vehicles. Can be increased if demand exceeds capacity
        data: data point and demand. Shape: [K, 3] with K is number of data point. Each row is (lat, lon, demand)
        capacity : vehicle capacity
    """
    cluster_center, cluster_labels = None, None
    flag = True
    while flag:
        print(f"---------------Starting clustering with {num_cluster} vehicles---------------")
        ## Clustering
        my_model = FCM(n_clusters=num_cluster, max_iter=500) 
        my_model.fit(data[:,:2]) 
        cluster_center = my_model.centers
        cluster_labels = my_model.predict(data[:,:2])
        # cluster_center, weight_matrix, cluster_labels = fuzzy_c_means_clustering(num_cluster, data[:,:2])
        ## Check demand in each clusters
        is_exceeds = False
        if increase_vehicle:
            for i in range(num_cluster):
                demand = np.sum(data[cluster_labels == i][:,2:3])
                if demand > capacity:
                    print(f"Demand exceed at cluster {i} ({demand} > {capacity}). Increaseing number of vehicle to 1")
                    is_exceeds = True
                    num_cluster = num_cluster + 1
                    break
        if not is_exceeds:
            flag = False
            print(f"Finishing clustering with number of vehicles {num_cluster}")
    
    ## Formatting data
    distance_data = np.append(data.copy(), np.zeros((data.shape[0], 2)), axis = 1) # [x, y, demand, centroid_index, distance_to_centroid]
    distance_data[:, 3] = cluster_labels
    distance_data[:, 4] = np.sqrt(np.sum(((cluster_center[cluster_labels] - distance_data[:,0:2]) ** 2), axis=1))

    return cluster_center, distance_data


def plot_points(cluster_labels:np.ndarray, cluster_center:np.ndarray, data:np.ndarray):
    figure(figsize=(14, 12), dpi=80)
    cluster_label_unique_len = len(np.unique(cluster_labels))
    cluster_color = np.random.rand(cluster_label_unique_len, 3)  

    ## Plot point
    point_color = cluster_color[cluster_labels]
    for i in range(data.shape[0]):
        point = data[i]
        plt.scatter(point[0], point[1], marker="o", c=[point_color[i]])

     ## plot cluster center
    plt.scatter(cluster_center[:,0], cluster_center[:,1], marker="x", c=[[0,0,0]])

    plt.show()