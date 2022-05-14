function [distance]=distance_find(X,centroids,k,i) 
    %find distance between ponit and centroids 
    %vector distance=[dis_form cluster1,dis_from cluster...,dis_from clusterk] 
    for j=1:k 
        distance(j)=sqrt((X(i,1)-centroids(j,2))^2+(X(i,2)-centroids(j,3))^2); 
    end 
    end 
    