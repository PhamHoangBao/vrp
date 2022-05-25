import numpy as np
import math
import os 
import sys
from sklearn.cluster import KMeans

"""
    Inverse distance - Get an array of inverted distances
    @arg
        {numpy.ndarray} space   -- The space

    @return
        {numpy.ndarray}         -- A space.dimension per space.dimension array of inverse distances
"""
def inverse_distances(space):
    # Empty multidimensional array (matriz) to distances
    distances = np.zeros((space.shape[0], space.shape[0]))

    # Calculate distance to all nodes to all nodes
    for index, point in enumerate(space):
        distances[index] = np.sqrt(((space - point) ** 2).sum(axis = 1))

    # Floating-point error handling - Setted to known state
    with np.errstate(all = 'ignore'):
        # Invert the distances
        inv_distances = 1 / distances

    # Replace infinity by zero to prevent zero division error
    inv_distances[inv_distances == np.inf] = 0

    # Eta algorithm result, inverted distances
    return distances, inv_distances

def get_centroid_cluster_num(citynum):
    if citynum > 125:
        return math.floor(citynum / 25)
    elif citynum < 4:
        return citynum
    else:
        return 4

"""
    Initialize ants - Get an array of random initial positions of the ants in space
    @arg
        {numpy.ndarray} space   -- The space
        {int} colony            -- Number of ants in the colony

    @return
        {numpy.ndarry}          -- An array of indexes of initial positions of ants in the space
"""
def initialize_ants(space, colony):
    # Indexes of initial positions of ants
    return np.random.randint(space.shape[0], size = colony)

def calculate_Euclidean_distance(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)

def calculate_cost_fitness(path : list, space : np.ndarray):
    rs = 0
    path_len = len(path)
    for i in range(0, path_len - 1):
        rs = rs + calculate_Euclidean_distance(space[path[i]], space[path[i + 1]])
    rs = rs + calculate_Euclidean_distance(space[path[path_len - 1]], space[path[0]])
    return rs

def perform_two_opt_algorithm(path:np.ndarray, space:np.ndarray):
    rs_path = path.copy()
    for i in range(len(rs_path) - 3):
        if calculate_Euclidean_distance(space[rs_path[i]], space[rs_path[i + 1]]) + calculate_Euclidean_distance(space[rs_path[i + 2]], space[rs_path[i + 3]])\
            > calculate_Euclidean_distance(space[rs_path[i]], space[rs_path[i + 2]]) + calculate_Euclidean_distance(space[rs_path[i + 1]], space[rs_path[i + 3]]):
            # print(i)
            temp = rs_path[i + 1]
            rs_path[i + 1] = rs_path[i + 2]
            rs_path[i + 2] = temp
    return rs_path

def convert_path_to_matrix(path : np.ndarray, space : np.ndarray):
    ## This will return matrix that A[i,j] = 1 if i,j in on tour of path ##
    rs = np.zeros((space.shape[0], space.shape[0]))
    for i in range(len(path) - 1):
        rs[path[i], path[i + 1]] = 1
        rs[path[i + 1], path[i]] = 1
    return rs

def initiallize(space, pop_size, seperation_factor):
    ## Calculate Euclidean distance for the cities and get the number of cities as citynum ##
    citynum = space.shape[0]
    # inverted distance and distance
    distances, inverted_distance = inverse_distances(space)

    ## Initialize pheromone ##
    pheromones = np.ones((space.shape[0], space.shape[0]))

    # ## Number of ants ##
    # ants_colony = 15

    ## Apply k-means clustering for the cities ## Equation 7 to 10 (using Kmeans library)
    # define number of centorids
    k = get_centroid_cluster_num(citynum) 
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=500).fit(space)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    centroid_distance = kmeans.transform(space) ## Get distance from each city to centroid
    data = np.append(space.copy(), np.expand_dims(cluster_labels, axis=1), axis=1) # x, y, cluster_no
    data = np.append(data.copy(), np.expand_dims(np.min(centroid_distance, axis=1), axis=1), axis=1) # x, y, cluster_no, distance_to_centroid

    ## Seperate non-classified cities ## ## Equation 11
    for i in range(cluster_centers.shape[0]):
        cluster = cluster_centers[i]
        element_index = np.where(data[:, 2] == i)
        element = data[element_index][:, 3]
        element_std = np.std(element)
        element_mean = np.mean(element)
        for i in element_index[0]:
            element_distance = data[i, 3]
            if element_std != 0 and element_distance - element_mean >= seperation_factor * element_std:
                data[i, 2] = -1 # Define this city is classless
    # define city class relation
    class_relation = np.expand_dims(data[:,2], axis = 1) - np.expand_dims(data[:,2], axis = 0)
    class_relation = (class_relation == 0).astype(int)
    class_relation = np.where(class_relation == 0, -1, class_relation)
    classless_indices = np.where(data[:, 2] == -1)
    for classless_index in classless_indices:
        class_relation[classless_index, :] = 0
        class_relation[:, classless_index] = 0
    for i in range(space.shape[0]):
        class_relation[i,i] = 0

    ## Initialize ant position ##
    ant_positions = initialize_ants(space, pop_size)
    ant_positions.dtype = np.int32

    return ant_positions, class_relation, pheromones, inverted_distance

def run_ahaco(
        space : np.ndarray,
        t_max = 1000,  # Number of interation
        pop_size = 10, # Population size (number of ant)
        tries = 100,
        alpha = 1.0,   # alpha
        beta = 3.0,    # beta 
        seperation_factor = 1.5,  # Seperation factor
        max_reward_punish_factor = 8, # Maximum reward-punish factor
        p = 0.9,   # Evaporation coeIcient
        Q = 120    # total ammount of pheronome
        ):
    ant_positions, class_relation, pheromones, inverted_distance = initiallize(space, pop_size, seperation_factor)
    
    ## Define ant_path: this is optimal solution for each ant
    citynum = space.shape[0]

    ant_path = np.zeros((pop_size, citynum), dtype=np.int32)
    last_best_ant_path = np.zeros((pop_size, citynum), dtype=np.int32)
    fitness_array = np.zeros((pop_size, 1))
    best_ant_path = None
    best_fitness = float('inf')
    times = 0
    delta_reward_punish_factor = 2 * (max_reward_punish_factor - 1) / t_max
    reward_punish_factor = max_reward_punish_factor
    probability_matrix = np.zeros((space.shape[0], space.shape[0], pop_size)) # (order, city_list, ant_size))

    for iteration in range(1, t_max):
        if iteration < (t_max / 2):
            reward_punish_factor = reward_punish_factor - delta_reward_punish_factor
            Y = -1
        else:
            reward_punish_factor = reward_punish_factor + delta_reward_punish_factor
            Y = 1
        
        for k in range(pop_size): # Travesre in all ants
            location_prob = probability_matrix[:, :, k] # Get location matrix of ith ant
            # print(f"Location prob {location_prob.shape}")
            # print(f"process ant number {k}")
            for j in range(citynum): # Traverse cities in order ()
                # print(f"traverse {j}")
                if j == 0: # process Starting point
                    # print("starting point")
                    ant_starting_position = ant_positions[k]
                    one_hot = np.zeros(citynum)
                    one_hot[ant_starting_position] = 1
                    location_prob[j] = one_hot
                    ant_path[k][j] = ant_starting_position
                else:
                    allowed_k = np.ones(citynum)
                    allowed_k[ant_path[k, 0:j]] = 0
                    last_location =  ant_path[k][j - 1]

                    if k % 2 == 1 : ## Special ant
                        ## Calculate with equation (13)
                        next_location_prob = (pheromones[last_location, :] * (inverted_distance[last_location, :] ** 3) * (reward_punish_factor ** (Y * class_relation[last_location, :]))) * allowed_k
                        next_location_prob = next_location_prob / np.sum(next_location_prob)
                        location_prob[j] = next_location_prob
                        ## Getting nearest city index base on next_location_prob and assign it to next location
                        nearest_city = np.argmax(next_location_prob)
                        ant_path[k, j] = nearest_city
                        # break
                        # pass
                    else: ## Normal ant
                        ## Calculate with equation (2)
                        next_location_prob = (pheromones[last_location, :] ** alpha) * (inverted_distance[last_location, :] ** beta) * allowed_k 
                        next_location_prob = next_location_prob / np.sum(next_location_prob)
                        location_prob[j] = next_location_prob
                        ## Getting nearest city index base on next_location_prob and assign it to next location
                        nearest_city = np.argmax(next_location_prob)
                        ant_path[k, j] = nearest_city
                        # break
            # Calculate fitness of the corresponding solution obtained by ant i (equation (1))
            fitness_array[k] = calculate_cost_fitness(path=ant_path[k,:].tolist(), space=space)

            # break

        ## Select the best solution of all ants AND update global best solution (optimal solution)##
        best_ant_path = ant_path[np.argmin(fitness_array)]
        best_fitness = np.amin(fitness_array)

        ## Apply improved 2-opt algorithm to optimal solution (Section 3.3) ##
        if len(best_ant_path) >= 4:
            best_ant_path = perform_two_opt_algorithm(best_ant_path, space)
            best_fitness = calculate_cost_fitness(best_ant_path.tolist(), space)
        # print(f"iteration {iteration} get fitness {best_fitness}")  ## Fitness is optimal path length
        
        ## Update pheromone on normal and special best solution separately (equations (14)–(16)) ##
        min_normal_fitness_cost = fitness_array[2 * np.argmin(fitness_array[0::2])]
        min_special_fitness_cost = fitness_array[2 * np.argmin(fitness_array[1::2]) + 1]
        
        location_matrix = convert_path_to_matrix(best_ant_path, space)
        normal_delta_pheromone = location_matrix.copy() * (Q / min_normal_fitness_cost)
        special_delta_pheromone = location_matrix.copy() * (Q / min_special_fitness_cost)
        delta_pheromone = normal_delta_pheromone + special_delta_pheromone

        pheromones = (1 - p) * pheromones + delta_pheromone

        ## If global best solution is not updated ##
        if np.array_equal(best_ant_path, last_best_ant_path):
            times = times + 1
            if times > tries:
                # Re-initialize the pheromone on the global best solution
                pheromones = np.ones((space.shape[0], space.shape[0]))
                times = 0
        else:
            times = 0
        last_best_ant_path = best_ant_path
    return best_ant_path, best_fitness