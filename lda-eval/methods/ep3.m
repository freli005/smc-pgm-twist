function [lZ,beta] = ep3(w, phi, m, maxIter)
% words,        w:   1 x n
% topics,       phi: K x V
% topic_prior,  m:   K x 1

% This EP implementation uses one factor per word in the document, without
% any sharing

debugging = false;


if(~exist('maxIter','var'))
    maxIter = 250;
end
tolerance=1e-5;

% Start by reducing the vocabulary based on the current document
n = length(w); % Number of words
%[w_unique,~,w] = unique(w,'sorted');
%nw = histc(w,1:n)'; % Number of occurences of each word
%phi = phi(:,w_unique);
[K,V] = size(phi);

% Initialize the EP parameters (below Eq (16) of minka-aspect)
beta = zeros(K,n);
logs = zeros(1,n); % s initialized to 1
g = m; % Gamma

% EP loop
converged = false;
numIter = 0;
beta_old = beta;

while(~converged)
    for(j=n:-1:1) % Reverse order will "encourage" positive g's
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
            
            betap = gp-gm; % Harder constraint, implying  all gp's positive
            beta_tmp = beta;
            beta_tmp(:,j) = betap;
            validityCheck = all(all( bsxfun(@plus, m, cumsum(beta_tmp(:,end:-1:1),2)) >=0)) && all(gp >=0);
%             validityCheck = gp-gm; 
            if(validityCheck)
                g = gp; % Current posterior approximation
                beta(:,j) = betap;
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
    converged = max( abs(beta(:)-beta_old(:))./abs(beta_old(:))) < tolerance || (numIter >= maxIter);
    beta_old = beta;
end

lZ = sum(logs) + gammaln(sum(m))-gammaln(sum(g)) + sum(gammaln(g)-gammaln(m));
numIter