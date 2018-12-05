function [lZ,beta] = ep_minka(doc, phi, m, maxIter)
% document,     doc:   1 x n
% topics,       phi: K x V
% topic_prior,  m:   K x 1

% This EP implementation uses one factor per word in the document, but
% shared for same words

debugging = false;

if(~exist('maxIter','var'))
    maxIter = 250;
end
tolerance = 1e-5;

% Start by reducing the vocabulary based on the current document
n = length(doc); % Number of words
[w_unique,~,doc] = unique(doc,'sorted');
phi = phi(:,w_unique);
[K,V] = size(phi);
nw = histc(doc,1:V)'; % Number of occurences of each word

% Initialize the EP parameters (below Eq (16) of minka-aspect)
beta = zeros(K,V);
logs = zeros(1,V); % s initialized to 1
g = m; % Gamma

% EP loop
converged = false;
numIter = 0;
beta_old = beta;

while(~converged)
    for(w = 1:V)
        %%% Deletion
        gm = g - beta(:,w);
        
        if(all(gm >= 0)) % Only continue with this word if we get valid paramters
            %%% Moment matching
            gmSum = sum(gm);
            gphiSum = sum(gm.*phi(:,w));
            Zj = gphiSum/gmSum; % Normalizing constant for current factor
            mu = gm.*(phi(:,w) + gphiSum)/Zj/gmSum/(1+gmSum);
            mu2 = gm.*(gm+1).*(2*phi(:,w)+gphiSum)/Zj/gmSum/(1+gmSum)/(2+gmSum);
            gp = sum(mu-mu2)/sum(mu2-mu.^2)*mu;
            
            %%% Update
            if(all(gp >= 0))
                g = gp; % Current posterior approximation
                %beta(:,w) = gp-gm;
                beta(:,w) = 1/nw(w)*(gp-gm) + (1-1/nw(w))*beta(:,w);
                logs(w) = log(Zj) + gammaln(sum(gp))-gammaln(sum(gm)) + sum(gammaln(gm)-gammaln(gp));
            end
        end
    end
    
    %%% DEBUG
    if(debugging)
        figure(1);
%         subplot(2,1,1)
%         plot(numIter,g,'.'); hold on;
%         subplot(2,1,2)
%         plot(numIter,logs,'.'); hold on;
        semilogy(numIter, max( abs(beta(:)-beta_old(:))./abs(beta_old(:))),'.'); hold on;
         drawnow;
    end
    % Check convergence
    numIter = numIter+1;
    converged = max( abs(beta(:)-beta_old(:))./abs(beta_old(:))) < tolerance || (numIter >= maxIter);
    beta_old = beta;
end

numIter

% NC estimate
lZ = sum(nw.*logs) + gammaln(sum(m))-gammaln(sum(g)) + sum(gammaln(g)-gammaln(m));

% Return beta on the same format as other methods, where we have one factor
% per word in document
beta = beta(:,doc);



%%% DEBUG
%g - (m+sum(bsxfun(@times, nw, beta),2))


