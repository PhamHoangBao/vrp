function fitness=tourlength(centroids,A,n,k,distance,Vehicle_cap) 
    fitness=0; 
    total_cap=0; 
    centroids(:,4)=0; 
    centroids(:,5)=0; 
    for u=1:n 
        distance(u,2:k+1)=distance_find(A(:,2:3),centroids,k,u); 
    end 
    [distance,centroids]=cluster_find(distance,centroids,A,k,n,Vehicle_cap); 
    for u=1:k 
        total_cap=total_cap+max(0,centroids(u,4)-Vehicle_cap); 
        fitness=fitness+centroids(u,5); 
    end 
    fitness=fitness+1500*(total_cap/Vehicle_cap)^2; 
    