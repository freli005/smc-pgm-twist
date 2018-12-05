function [X,W,lZ,alpha_log,ess_log] = smcsampler(Jv, Jh, H, N, alpha, par)
% SMC Sampeler (Annealed SIR) for Ising model, nx x ny. Can handle periodic or free boundary
% conditions.
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
% alpha, 1 x (n+1): Annealing sequence, where n is the number of annealing
%                  steps. alpha(1) = 0, alpha(n+1) = 1
%       []: Adaptive annealing
% par.resampling: 0 = off (AIS), 1 = multinomial, 2 = stratified, 3 = systematic
%    .ESS_threshold: Value in [0,1], determining when to resample
%    .CESS_threshold: Value in [0,1], CESS target value used when adapting alpha

par.method = 'asir';
adaptive = numel(alpha)==0;

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


%"Sample" uniformly, stratified sample on discrete distribution
% (This is assuming that we anneal the external field as well)
X = zeros(nx,ny,N);
for(iX = 1:nx)
    for(iY = 1:ny)
        tmpVec = [-ones(1,floor(N/2)) ones(1,ceil(N/2))];
        
        X(iX,iY,:)=tmpVec(randperm(N)); 
    end
end
lZ0 = nx*ny*log(2);

% Run sampler, adaptive or deterministic schedule
if(adaptive)
    [X,lZ,logW,alpha_log,ess_log] = asir_adpt(X,0,0,0,Jv,Jh,H,0,par);
    alpha_log = cumsum(alpha_log);
else
    [X,lZ,logW,ess_log] = asir(X,0,0,0,Jv,Jh,H,alpha,par);
    alpha_log = alpha;
end

% Compute final weights and lZ
maxlW = max( logW );
w = exp( logW - maxlW );
lZ = lZ0 + lZ + maxlW + log( sum( w ) ) - log( N );
W = w/sum(w);
end