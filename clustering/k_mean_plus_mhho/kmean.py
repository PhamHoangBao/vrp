import numpy as np
import pandas as pd
import math
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import random
import time

def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta) 
    u= 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step

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

def find_cluster(centroids, data, vehicle_cap, p, c):
    distance_matrix = calculate_distance_matrix(centroids, data[:, :2])
    new_centroids = np.zeros((centroids.shape[0], 5))
    for i in range(centroids.shape[0]):
        new_centroids[i, 0] = i 
    new_centroids[:, 1:3] = centroids
    
    distance_data = np.append(data.copy(), np.zeros((data.shape[0], 2)), axis = 1) # [x, y, demand, centroid_index, distance_to_centroid]
    capacity_violation = 0
    ## Assign to nearest cluster and update centroid capacity
    for i in range(data.shape[0]):
        # print(f"process data {i} : {data[i, :]}")
        centroid_distance = distance_matrix[i, :]
        nearest_centroid_order = np.argsort(centroid_distance)
        is_violated = True
        for nearest_centroid_idx in nearest_centroid_order:
            ## Check adding customer demand exceeds vehicle capacity 
            if (new_centroids[nearest_centroid_idx, 3] + data[i, 2]) < vehicle_cap:
                # print(f"data {i} is not violate. Insert to centroid {nearest_centroid_idx}")
                new_centroids[nearest_centroid_idx, 3] = new_centroids[nearest_centroid_idx, 3] + data[i, 2]
                is_violated = False
                distance_data[i, 3] = nearest_centroid_idx
                distance_data[i, 4] = centroid_distance[nearest_centroid_idx]
                break
        if is_violated: # If all capacity is violate, assign to cluster have minimum violation (minimum current capacity)
            min_cap_centroid_idx = np.argmin(new_centroids[:, 3], axis = 0)
            new_centroids[min_cap_centroid_idx, 3] = new_centroids[min_cap_centroid_idx, 3] + data[i, 2]
            capacity_violation = capacity_violation + new_centroids[min_cap_centroid_idx, 3] - vehicle_cap
            distance_data[i, 3] = min_cap_centroid_idx
            distance_data[i, 4] = centroid_distance[nearest_centroid_idx]
            # print(f"data {i} is violated. Insert to neareste centroid with minimum violation {min_cap_centroid_idx}")
    fitness = calculate_fitness(distance_data, capacity_violation, p, c)
    return new_centroids, distance_data, capacity_violation, fitness

def calculate_fitness(distance_data, capacity_violation, p, c):
    return np.sum(distance_data[:,4] ** 2) + p * capacity_violation ** c

def plot_points(centroid : np.ndarray, distance_data : np.ndarray):
    figure(figsize=(14, 12), dpi=80)
    cluster_color = np.random.rand(centroid.shape[0], 3)
    # Plot point  
    for i in range(distance_data.shape[0]):
        data = distance_data[i]
        # print(cluster_color[data[3]])
        plt.scatter(data[0], data[1], marker="o", c = [cluster_color[int(data[3])]])
    # Plot cluster center
    plt.scatter(centroid[:,1], centroid[:,2], marker="x", c=[[0,0,0]])

def run_kmean_mhho(
        data : np.ndarray,
        num_cluster = 13,  ## Number of cluster ##
        vehicle_capacity = 66, ## Violate capacity ##
        p = 1500,  ## Penalty number
        c = 2,
        SearchAgents_no = 30 ## Number of search agent ##
    ):

    ## Data matrix ##
    data_matrix = data[:,:2] 
    ## Demanbd vector ##
    demand_vector = data[:,2:]  
    ## number of data ##
    n = data.shape[0]
    ## Define centroids ##
    centroids = np.zeros((num_cluster, 5)) ## Centroid matrix [no.Cluster, x, y, current capacity, total distance]
    # distance = np.zeros(n, num_cluster + 2)
    ## Maximum iteration ##
    max_iter = 500
    convergence_curve = np.zeros(max_iter)
    ## Lower bound ##
    lb = np.array([np.min(data_matrix[:,0]), np.min(data_matrix[:,1])])
    # lb = 0
    ## Upper bound ##
    ub = np.array([np.max(data_matrix[:,0]), np.max(data_matrix[:,1])])
    # ub = 100
    ## Fitness matrix ##

    ### Define no. cluster ##
    for i in range(num_cluster):
        centroids[i, 0] = i 

    dim = 2  ## Each point contain only x and y coordinates
    t = 0
    # Initialize the locations of Harris' hawks (position of search agents)
    X = np.random.uniform(0, 1, (SearchAgents_no, num_cluster, dim)) * (ub - lb) + lb
    # Initialize the location and Energy of the rabbit
    Rabbit_Location = np.zeros((num_cluster, dim))
    # Centroid_Location = 
    Rabbit_Energy = float("inf")  #change this to -inf for maximization problems
    ## Fitness value ##
    fitness_values = np.zeros(SearchAgents_no)

    while t < max_iter:
        print(f"Iteration {t}")
        start_time = time.time()
        for i in range(0, SearchAgents_no):
            # Check boundaries
            X[i, :] = np.clip(X[i, :], lb, ub)
            # Fitness of location
            new_centroids, distance_data, capacity_violation, fitness = find_cluster(X[i, :].copy(), data, vehicle_capacity, p, c)
            
            # Update the location of Rabbit
            if fitness < Rabbit_Energy: # Change this to > for maximization problem
                print(f"Update fitness at {i}")
                Rabbit_Energy = fitness 
                Rabbit_Location = X[i, :].copy() 
                centroids = new_centroids
        E1 = 2 * (1 - (t / max_iter)) # factor to show the decreaing energy of rabbit    
        # print(f"Process A : {time.time() - start_time}")
        # Update the location of Harris' hawks 
        start_time = time.time()
        for m in range(0, SearchAgents_no):
            E0 = 2 * random.random() - 1;  # -1<E0<1
            Escaping_Energy = E1 * (E0)  # escaping energy of rabbit Eq. (3) in the paper

            # for u in range(num_cluster):
            # -------- Exploration phase Eq. (1) in paper -------------------
            # m = i * num_cluster + u
            # m = i
            if abs(Escaping_Energy) >= 1:
                #Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5:
                    # perch based on other family members
                    X[m,:] = X_rand - random.random() * abs(X_rand - 2 * random.random() * X[m,:])

                elif q >= 0.5:
                    #perch on a random tall tree (random site inside group's home range)
                    X[m,:] = (Rabbit_Location - X.mean(axis=0)) - random.random() * ((ub - lb) * random.random() + lb)

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy) < 1:
                #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                #phase 1: ----- surprise pounce (seven kills) ----------
                #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random() # probablity of each event
                
                if r >= 0.5 and abs(Escaping_Energy) < 0.5: # Hard besiege Eq. (6) in paper
                    X[m,:] = (Rabbit_Location) - Escaping_Energy * abs(Rabbit_Location - X[m,:])

                if r >= 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft besiege Eq. (4) in paper
                    Jump_strength = 2 * (1 - random.random()); # random jump strength of the rabbit
                    X[m,:] = (Rabbit_Location - X[m,:]) - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[m,:])
                
                #phase 2: --------performing team rapid dives (leapfrog movements)----------

                if r < 0.5 and abs(Escaping_Energy) >= 0.5: # Soft besiege Eq. (10) in paper
                    #rabbit try to escape by many zigzag deceptive motions
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[m,:])
                    
                    new_fitness_X1 = find_cluster(X1, data, vehicle_capacity, p, c)[3]
                    # if objf(X1) < fitness: # improved move?
                    if new_fitness_X1 < fitness: # improved move?
                        X[m,:] = X1.copy()
                    else: # hawks perform levy-based short rapid dives around the rabbit
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[m,:]) + np.multiply(np.random.randn(dim), Levy(dim))
                        new_fitness_X2 = find_cluster(X2, data, vehicle_capacity, p, c)[3]
                        # if objf(X2) < fitness:
                        if new_fitness_X2 < fitness:
                            X[m,:] = X2.copy()
                if r < 0.5 and abs(Escaping_Energy) < 0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength = 2 * (1 - random.random())
                        X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(0))

                        new_fitness_X1 = find_cluster(X1, data, vehicle_capacity, p, c)[3]
                        # if objf(X1) < fitness: # improved move?
                        if new_fitness_X1 < fitness: # improved move?
                            X[m,:] = X1.copy()
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(0)) + np.multiply(np.random.randn(dim), Levy(dim))
                            new_fitness_X2 = find_cluster(X2, data, vehicle_capacity, p, c)[3]
                            # if objf(X2) < fitness:
                            if new_fitness_X2 < fitness:
                                X[m,:] = X2.copy() 

        convergence_curve[t] = Rabbit_Energy
        if (t % 1 == 0):
                print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Rabbit_Energy)])
        t = t + 1
        # print(f"Process B : {time.time() - start_time}")

    return centroids, distance_data