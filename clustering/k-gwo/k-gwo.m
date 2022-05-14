function [total_tour_length,cap_violate,Alpha_pos]=Kmeans(v) 
    [A,k,Vehicle_cap]=text_read(v); 
    X=A(:,2:3); %data matrix 
    Y=A(:,4); %demand vector 
    n=size(A,1); 
    opts=statset('Display','final'); 
    centroids=zeros(k,5);%centroids matrix[ no.cluster, x, y, current capacity,total distance] 
    distance=zeros(n, k+2); %distance matrix[no.point, cluster 1...cluster k, cluster] 
    distance(:,1)=A(:,1); 
    for i=1:k 
        centroids(i,1)=i; 
    end 
    SearchAgents_no=30; 
    dim=2; 
    lb=[min(X(:,1)), min(X(:,2))]; 
    ub=[max(X(:,1)), max(X(:,2))]; 
    Max_iter=500; 
    % initialize alpha, beta, and delta_pos 
    Alpha_pos=zeros(k,dim); 
    Alpha_score=inf; %change this to -inf for maximization problems 
    Beta_pos=zeros(k,dim); 
    Beta_score=inf; %change this to -inf for maximization problems 
    Delta_pos=zeros(k,dim); 
    Delta_score=inf; %change this to -inf for maximization problems 
    %Initialize the positions of search agents 
    Positions=initialization(SearchAgents_no*k,dim,ub,lb); 
    Convergence_curve=zeros(1,Max_iter); 
    l=0;% Loop counter 
    fitness=zeros(1,SearchAgents_no); 
    while (l<Max_iter) 
        fprintf('%d - %d \n',v,l); 
        for i=1:size(Positions,1) 
            % Return back the search agents that go beyond the boundaries of the search space 
            Flag4ub=Positions(i,:)>ub; 
            Flag4lb=Positions(i,:)<lb; 
            Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; 
        end 
        for i=0:SearchAgents_no-1 
            centroids(:,2:3)=define_cen(Positions,i,k); 
            fitness(i+1)=tourlength(centroids,A,n,k,distance,Vehicle_cap); 
        end 
        a1=zeros(1,3); 
        a_index=zeros(1,3); 
        for i=1:3 
            [minfit,idx]=min(fitness); 
            if i==1 
                a1(1)=minfit; 
                a_index(1)=idx; 
            elseif i==2 
                a1(2)=minfit; 
                a_index(2)=idx; 
            elseif i==3 
                a1(3)=minfit; 
                a_index(3)=idx; 
        end 
        fitness(idx)=inf; 
        end 
        for i=1:3 
            if a1(i)<Alpha_score 
                Alpha_score=a1(i);% Update alpha 
                Alpha_pos=define_cen(Positions,a_index(i)-1,k); 
            elseif a1(i)<Beta_score 
                Beta_score=a1(i);% Update alpha 
                Beta_pos=define_cen(Positions,a_index(i)-1,k); 
            elseif a1(i)<Delta_score 
                Delta_score=a1(i);% Update alpha 
                Delta_pos=define_cen(Positions,a_index(i)-1,k); 
            end 
        end 
        a=2-l*((2)/Max_iter); % a decreases linearly fron 2 to 0 
        % Update the Position of search agents including omegas 
        for i=0:SearchAgents_no-1 
            cen=define_cen(Positions,i,k); 
            for u=1:k 
                for j=1:size(Positions,2) 
                    r1=rand(); % r1 is a random number in [0,1] 
                    r2=rand(); % r2 is a random number in [0,1] 
                    A1=2*a*r1-a; % Equation (3.3) 
                    C1=2*r2; % Equation (3.4) 
                    D_alpha=abs(C1*Alpha_pos(u,j)-cen(u,j)); % Equation (3.5)-part 1 
                    X1=Alpha_pos(u,j)-A1*D_alpha; % Equation (3.6)-part 1 
                    r1=rand(); 
                    r2=rand(); 
                    A2=2*a*r1-a; % Equation (3.3) 
                    C2=2*r2; % Equation (3.4) 
                    D_beta=abs(C2*Beta_pos(u,j)-cen(u,j)); % Equation (3.5)-part 2 
                    X2=Beta_pos(u,j)-A2*D_beta; % Equation (3.6)-part 2 
                    r1=rand(); 
                    r2=rand(); 
                    A3=2*a*r1-a; % Equation (3.3) 
                    C3=2*r2; % Equation (3.4) 
                    D_delta=abs(C3*Delta_pos(u,j)-cen(u,j)); % Equation (3.5)-part 3 
                    X3=Delta_pos(u,j)-A3*D_delta; % Equation (3.5)-part 3 
                    cen(u,j)=(X1+X2+X3)/3;% Equation (3.7) 
                end 
            end 
            for u=1+i*k:k+i*k 
                Positions(u,:)=cen(u-i*k,1:2); 
            end 
        end 
        l=l+1; 
        Convergence_curve(l)=Alpha_score; 
        if (l==100)||(l==500) 
            centroids(:,2:3)=Alpha_pos; [total_tour_length,cap_violate]=total_distance(centroids,k,n,A,Vehicle_cap); 
            tsp_curve(l)=total_tour_length; 
            cap_curve(l)=cap_violate; 
        end 
        end 
    end 
    