function lZ = fapf_twist(w, phi, m, ep_posterior, Np)
% words,        w:   1 x n
% topics,       phi: T x V
% topic_prior,  m:   T x 1

% Topic prior
%alpha = sum(m);
%m = m/alpha;

% Topic posterior from EP (ep2); Note that m is not normalized here
%g = m+sum(beta(:,2:end),2); % At iteration k=1 we use the EP factors from 2...n
g = ep_posterior(:,2);

n = length(w);
T = length(m);

z = zeros(Np,n);

% Initialize
gm = phi(:,w(1)).*g/sum(g);
lZ = log(sum(gm));
z(:,1) = discreternd(Np, gm/sum(gm));

% Count number of occurences of each word
tau = zeros(Np,T);
ti = sub2ind([Np T], (1:Np)', z(:,1));
tau(ti) = 1;

% Keep track of normalized Dirichlet "prior"
c = log( g(z(:,1))/sum(g) );

for k = 2:n
    % Update "prior" by removing one term
    %g = g-beta(:,k);
    if(k < n)
        g = ep_posterior(:,k+1);
    else
        g = m;
    end
    ghat = sum(g);
    
    % Include next word, full adaptation (compute for all possible values
    % of z_t first
    
    %gm = ( bsxfun(@times, phi(:,w(k)), bsxfun(@plus, tau',
    %alpha*m))/(k-1+alpha) )'; % <- no twisting
    
    % This is the part of the constant that does not depend on z_t;
    % dimension here is [Np 1]
    C = gammaln(ghat)-gammaln(ghat+k) + sum(gammaln(bsxfun(@plus, tau, g')),2)-sum(gammaln(g));
     % Here C is expanded in second dimension and becoms [Np K]
    C = bsxfun(@plus, C, log(bsxfun(@plus, tau, g')));
    
    % Ratio of targets: C [Np K], c [Np 1], phi' [1 K]
    gm = bsxfun(@times, exp( bsxfun(@minus, C, c) ), phi(:,w(k))');
    
    nu = sum(gm,2);
    lZ = lZ + log(sum(nu))-log(Np);
    % Resample (not ideal to resample at every time step, but ok..)
    ind = resampling(nu/sum(nu),3);
    tau = tau(ind,:);
    gm = gm(ind,:);
    nu = nu(ind);
    C = C(ind,:);
    % Propagate (this can be done in a better way!!)    
    gm = bsxfun(@rdivide, gm, nu);
    z_tmp = repmat(rand(Np,1),[1 T]) < cumsum(gm,2);
    z(:,k) = T+1-sum(z_tmp,2);
    ti = sub2ind([Np T], (1:Np)', z(:,k));
    tau(ti) = tau(ti)+1;
    
    c = C(ti); % Store constant of artificial prior for next iteration. Note that C is "resampled" first
end