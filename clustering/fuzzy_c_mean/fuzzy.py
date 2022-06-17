import numpy as np
from sklearn.cluster import KMeans
from fcmeans import FCM
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

### CALCULATE DISTANCE MATRIX ###
def calculate_distance_matrix(centroids, points):
    """
        centroids: shape(k,2)
        points: shape(m, 2)

        output : shape (m, k)
    """
    centroids_ext = np.expand_dims(centroids, axis=0)
    points_ext = np.expand_dims(points, axis=1)
    return np.sum((centroids_ext - points_ext) ** 2, axis=2) ** 0.5

def remove_violate_element(sort_centroids, distance_data, capacity):
    for sort_centroid in sort_centroids:
        if sort_centroid[3] > capacity: ## If cluster is violated
            centroid_index = sort_centroid[0]
            ## Get index of element that belong to cluster
            element_index = np.where(distance_data[:, 3] == centroid_index)
            ## Get element that belong to the cluster
            element = distance_data[distance_data[:, 3] == centroid_index]
            # print(element)
            ## Sort element by their demand (ascending)
            sort_element_indices = np.argsort(element[:,2])
            ## Loop for remove all violated element
            for sort_element_index in sort_element_indices:
                if sort_centroid[3] > capacity:
                    sort_centroid[3] = sort_centroid[3] - element[sort_element_index][2]
                    element[sort_element_index][3] = -1
                else:
                    break
            ## Update element data to distance data
            distance_data[element_index] = element
        else:
            break
    return sort_centroids, distance_data

def post_process(cluster_centroids : np.ndarray,
                distance_data : np.ndarray, 
                capacity : int):
    
    ### Calculate total demand in each clusters ###
    centroids = None  ## CLuster index, x, y, total demand
    frame = pd.DataFrame(distance_data)
    demands = frame.groupby(3).sum()[2].to_numpy()
    centroids = np.append(
                cluster_centroids.copy(), 
                np.array([demands]).T,
                axis=1)
    centroids = np.append(
                np.expand_dims(np.arange(cluster_centroids.shape[0]), axis=0).T,
                centroids,
                axis=1
                )
    ### REMOVE POINT THAT VIOLATE OUT OF CLUSTER ###
    sort_centroids = centroids[centroids[:,3].argsort()[::-1]]
    ## element data with -1 is not clustered 
    udpated_centroids, updated_distance_data = remove_violate_element(sort_centroids, distance_data.copy(), capacity)
    udpated_centroids = udpated_centroids[udpated_centroids[:,0].argsort()]
    ### ASSIGN UN_CLUSTERED ELEMENT TO NEW CLUSTER ###
    unclustered_data_indices= np.argwhere(updated_distance_data[:, 3] == -1)
    unclustered_distance_matrix = calculate_distance_matrix(udpated_centroids[:,1:3], updated_distance_data[:, :2])
    for unclustered_data_index in np.squeeze(unclustered_data_indices.T):
        ## Check adding customer demand exceeds vehicle capacity 
        unclustered_element = updated_distance_data[unclustered_data_index]
        centroid_distance = unclustered_distance_matrix[unclustered_data_index]
        nearest_centroid_order = np.argsort(centroid_distance)
        is_violated = True
        for nearest_centroid_idx in nearest_centroid_order:
            ## Check adding customer demand exceeds vehicle capacity 
            if udpated_centroids[nearest_centroid_idx, 3] + unclustered_element[2] <= capacity:
                udpated_centroids[nearest_centroid_idx, 3] = udpated_centroids[nearest_centroid_idx, 3] + unclustered_element[2]
                is_violated = False
                updated_distance_data[unclustered_data_index, 3] = nearest_centroid_idx
                break
        if is_violated:# If all capacity is violate, assign to cluster have minimum violation (minimum current capacity)
            min_cap_centroid_idx = np.argmin(udpated_centroids[:, 3], axis = 0)
            udpated_centroids[min_cap_centroid_idx, 3] = udpated_centroids[min_cap_centroid_idx, 3] + unclustered_element[2]
            updated_distance_data[unclustered_data_index, 3] = min_cap_centroid_idx

    return centroids, udpated_centroids, updated_distance_data   

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

    old_centroids, udpated_centroids, updated_distance_data = post_process(cluster_center, distance_data, capacity)

    return udpated_centroids, updated_distance_data


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