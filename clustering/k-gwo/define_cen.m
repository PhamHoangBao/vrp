function [centroid]=define_cen(Positions,i,k) 
    for u=1+i*k:i*k+k 
        centroid(u-i*k,1:2)=Positions(u,:); 
    end 
    