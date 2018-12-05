function [X, W, lZ, ess_log, Alog] = smcpgm(Jv, Jh, H, N, par)
% SMC for Ising model, sequential decomposition approach (L-R). Can handle
% periodic or free boundary conditions.
%
% Jv, M1 x M2: vertical interactions, Jv(i,j) is the interaction from
%            X(i,j) to X(i+1,j). If 'par.free_boundary = false', then the
%            M1'th row of Jv containts the boundary interactions between
%            X(M1,j) and X(1,j).
% Jh, M1 x M2: horizontal interactions, Jh(i,j) is the interaction from
%            X(i,j) to X(i,j+1). If 'par.free_boundary = false', then the
%            M2'th column of Jh contains the boundary interactions. 
% H, nx x ny: external field
% N, integer: number of particles
% par.resampling: 0 = off, 1 = multinomial, 2 = stratified, 3 = systematic
%    .ESS_threshold: Value in [0,1], determining when to resample

% Output:
%   X, nx x ny x N: final samples
%   lZ, scalar: estimate of log-normalizing constant
%   logW, N x 1: final unnormalised log-weights
%   ess_log, 1 x (n+1): ESS values at each iteration

par.method = 'smcpgm';

% For free boundary conditions, assume periodic with zero interaction
if(par.periodic)
    [nx,ny] = size(Jv); % Size of edge set (same for Jh)
    if(size(Jh,1)~=nx || size(Jh,2)~=ny)
        error('Wrong size of edge potentials!')
    end
else
    nx = size(Jv,1)+1;
    ny = size(Jh,2)+1;
    
    if(size(Jh,1)~=nx || size(Jv,2)~=ny)
        error('Wrong size of edge potentials!')
    end    

    Jv = [Jv ; zeros(1,ny)];
    Jh = [Jh zeros(nx,1)];
end

% Setup SMC
X = zeros(nx,ny,N);
logW = zeros(N,1);
ess_log = zeros(nx,ny);
lZ = 0;

% Logging degeneracy
Alog = zeros(nx,ny,N);

for iX = 1:nx % Loop over rows
    for iY = 1:ny % Loop over cols
        % Spos (1 x 1 x N) containts the unnormalized conditional
        % log-probability of X(iX,iY) = 1 for each particle (The model is
        % symmetric in the spin configurations in the sense that -Spos
        % gives the unnormalized conditional log-probability for X(iX,iY) =
        % -1). This is used to compute the fully adapted proposal below.
        
        % Local factor (from external magnetic field)
        Spos = H(iX,iY)*ones(1,1,N); 
               
        % Sum up all edge contributions coming from adding the new site
        if(iX > 1) % Above
            Spos = Spos + Jv(iX-1,iY)*X(iX-1,iY,:);
        end
        if(iX == nx && nx > 1) % Below (closing the periodic boundary)
            Spos = Spos + Jv(nx,iY)*X(1,iY,:);
        end
        if(iY > 1) % To the left
            Spos = Spos + Jh(iX,iY-1)*X(iX,iY-1,:);
        end
        if(iY == ny && ny > 1) % To the right (closing the periodic boundary)
            Spos = Spos + Jh(iX,ny)*X(iX,1,:);
        end

        % Assume resampling at every time step
        % logW = log( exp(Spos)+exp(-Spos) );
        idx = Spos > 0;
        logW(idx) = Spos(idx) + log( 1+exp(-2*Spos(idx)) );
        logW(~idx) = -Spos(~idx) + log( 1+exp(2*Spos(~idx)) );

        %%% Compute ESS and log-NC
        maxlW = max( logW );
        w = exp( logW - maxlW );
        W = w/sum(w);
        ess = 1/(N*sum(W.^2));
        % Store for analysis
        ess_log(iX,iY) = ess;
        
        % Log-normalizing constant estimate
        lZ = lZ + maxlW + log( sum( w ) ) - log( N );
        
        % Resample (unless first step)
        if( (iX > 1 || iY > 1) && par.resampling > 0)
            ind = resampling(W,par.resampling);
            X = X(:,:,ind);
            Alog = Alog(:,:,ind);
            Spos = Spos(ind);
        end
        
        % Propagate using optimal proposal
        Qpos = 1./(1+exp(-2*Spos)); % Probability of positive
        X(iX,iY,:) = 2*(rand(1,1,N) < Qpos)-1;
        Alog(iX,iY,:) = 1:N;
    
%     %% Resample if needed
%     if(iAlpha < numAlpha && par.resampling > 0 && ess < par.ESS_threshold) % Never resample at last iterations
%         ind = resampling(W,par.resampling);
%         X = X(:,:,ind);
%         Update normalizing constant at this iteration
%         lZ = lZ + maxlW + log( sum( w ) ) - log( N );
%         Reset weights
%         logW = zeros(N,1);
%     end
    end
end

%%% N.B. Since we use FA and resample at every time step, we get uniform
%%% weights. Change this if ESS-based resampling.
W = ones(N,1)/N;