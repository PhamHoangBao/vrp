function [distance,centroids]=cluster_find(distance,centroids,A,k,n,Vehicle_cap) 
    for i=1:n 
        [min_dis,index]=min(distance(i,2:k+1)); 
        distance(i,k+2)=index; 
    end 
        for i=1:n 
            r=distance(i,k+2); 
            if (A(i,4)+centroids(r,4)<=Vehicle_cap) 
                centroids(r,4)=A(i,4)+centroids(r,4); 
                centroids(r,5)=distance(i,r+1)+centroids(r,5); 
            else 
                [out,index]=sort(distance(i,2:k+1)); 
                l=2; 
                distance(i,k+2)=index(l); centroids(index(l),4)=A(i,4)+centroids(index(l),4); 
                centroids(index(l),5)=distance(i,index(l)+1)+centroids(index(l),5); 
                while (centroids(index(l),4)>Vehicle_cap)&&(l<=k-1) 
                    [out,index]=sort(distance(i,2:k+1)); 
                    distance(i,k+2)=index(l+1); 
                    centroids(index(l+1),4)=A(i,4)+centroids(index(l+1),4); 
                    centroids(index(l),4)=-A(i,4)+centroids(index(l),4); 
                    centroids(index(l+1),5)=distance(i,index(l+1)+1)+centroids(index(l+1),5); 
                    centroids(index(l),5)=-distance(i,index(l)+1)+centroids(index(l),5); 
                    l=l+1; 
                end 
                if (centroids(index(k),4)>Vehicle_cap) 
                    centroids(index(k),4)=-A(i,4)+centroids(index(k),4); 
                    centroids(index(k),5)=-distance(i,index(k)+1)+centroids(index(k),5); 
                    [min_cap,index]=min(centroids(:,4)); 
                    distance(i,k+2)=index; 
                    centroids(index,4)=A(i,4)+centroids(index,4); 
                    centroids(index,5)=distance(i,index+1)+centroids(index,5); 
                end 
            end 
        end 
    end 
    