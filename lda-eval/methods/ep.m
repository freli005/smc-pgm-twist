function lZ = ep(w, phi, m)
% words,        w:   1 x n
% topics,       phi: K x V
% topic_prior,  m:   K x 1

% This EP implementation uses one factor per word in the document, but
% shared for same words

debugging = false;

% Start by reducing the vocabulary based on the current document
n = length(w); % Number of words
[w_unique,~,w] = unique(w,'sorted');
phi = phi(:,w_unique);
[K,V] = size(phi);
nw = histc(w,1:V)'; % Number of occurences of each word

% Initialize the EP parameters (below Eq (16) of minka-aspect)
beta = zeros(K,V);
logs = zeros(1,V); % s initialized to 1
g = m; % Gamma

% EP loop
converged = false;
numIter = 0;
maxIter = 50;
while(~converged)
    for(j = 1:n)
        %%% Deletion
        gm = g - beta(:,w(j));
        
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
                beta(:,w(j)) = gp-gm;
                logs(w(j)) = log(Zj) + gammaln(sum(gp))-gammaln(sum(gm)) + sum(gammaln(gm)-gammaln(gp));
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

lZ = sum(nw.*logs) + gammaln(sum(m))-gammaln(sum(g)) + sum(gammaln(g)-gammaln(m));


%%% DEBUG
g - (m+sum(bsxfun(@times, nw, beta),2))


