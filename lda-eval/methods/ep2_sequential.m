function [lZ,glog] = ep2_sequential(w, phi, m, maxIter)
% words,        w:   1 x n
% topics,       phi: K x V
% topic_prior,  m:   K x 1

% This EP implementation uses one factor per word in the document, without
% any sharing. It runs EP "until convergence" for each word, to open up for
% using it with SMC.

debugging = false;

if(~exist('maxIter','var'))
    maxIter = 50;
end

% Start by reducing the vocabulary based on the current document
n = length(w); % Number of words
%[w_unique,~,w] = unique(w,'sorted');
%nw = histc(w,1:n)'; % Number of occurences of each word
%phi = phi(:,w_unique);
[K,V] = size(phi);

% Initialize the EP parameters (below Eq (16) of minka-aspect)
beta = zeros(K,n);
logs = zeros(1,n); % s initialized to 1
g = m;
glog = zeros(K,n); % Gamma
%lZ = zeros(1,n);

% EP loop
for(sj = n:-1:1) 
    converged = false;
    numIter = 0;

     % Reset?!
    %beta = zeros(K,n);
    %logs = zeros(1,n); % s initialized to 1

    
    while(~converged)
        for(j=sj:n)
            %%% Deletion
            gm = g - beta(:,j);
            
            if(all(gm >= 0)) % Only continue with this word if we get valid paramters
                %%% Moment matching
                gmSum = sum(gm);
                gphiSum = sum(gm.*phi(:,w(j)));
                Zj = gphiSum/gmSum; % Normalizing constant for current factor
                mu = gm.*(phi(:,w(j)) + gphiSum)/Zj/gmSum/(1+gmSum);
                mu2 = gm.*(gm+1).*(2*phi(:,w(j))+gphiSum)/Zj/gmSum/(1+gmSum)/(2+gmSum);
                gp = sum(mu-mu2)/sum(mu2-mu.^2)*mu;
                
                %%% Update
                if(all(gp >= 0))
                    g = gp; % Current posterior approximation
                    beta(:,j) = gp-gm;
                    logs(j) = log(Zj) + gammaln(sum(gp))-gammaln(sum(gm)) + sum(gammaln(gm)-gammaln(gp));
                end
            end
        end
        
        %%% DEBUG
        if(debugging)
            figure(1);
            subplot(2,1,1)
            plot(numIter,g,'.'); hold on;
            subplot(2,1,2)
            plot(numIter,logs,'.'); hold on;
            drawnow;
        end
        
        % Check convergence
        numIter = numIter+1;
        converged = (numIter >= maxIter);
    end
    glog(:,sj) = g;
    lZ = sum(logs) + gammaln(sum(m))-gammaln(sum(g(:,1))) + sum(gammaln(g(:,1))-gammaln(m));
end


%%% DEBUG
%g - (m+sum(beta,2))