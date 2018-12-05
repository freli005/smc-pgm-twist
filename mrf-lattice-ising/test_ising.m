% Proof of concept numerical illustration of Twisted Sequential Monte Carlo
% for inference in probabilistic graphical models. See,
%
%    F. Lindsten, J. Helske, and M. Vihola. Graphical model inference:
%    Sequential Monte Carlo meets deterministic approximations. 32nd
%    Conference on Neural Information Processing Systems (NeurIPS),
%    Montréal, Canada, December 2018.
%
% Task: estimate normalizing constant in a small square lattice Ising
% model. Please note that this implementation does not scale well with the
% size of the graph, due to poor memory management in the resampling step.
% A better way of implementing SMC for non-Markovian models (such as PGMs)
% is to use the path storage method by Jacob, Murray, and Rubenthaler
% https://arxiv.org/abs/1307.3180

addpath('./utils');
addpath('./methods');

%% Setup - Ising model with or without periodic boundary condition
par.periodic = true;
n = 1;
nx = 2^5;
ny = 2^5;
inverseTemperature = 1/2.26;

if(par.periodic)
    Jv = inverseTemperature*ones(nx,ny);
    Jh = inverseTemperature*ones(nx,ny);
else
    Jv = inverseTemperature*ones(nx-1,ny);
    Jh = inverseTemperature*ones(nx,ny-1);
end

H = 2*rand(nx,ny)-1; % External field
MC = 1;

% Method parameters
N = 512; % Number of particles
par.resampling = 3;
par.ESS_threshold = 0.5;
par.CESS_threshold = 0.995; % Used for adaptive annealing in ASIR/SMC Sampler
par.CESS_tolerance = 1e-5;

% Run
warning on;
par.plotOn = 0;
par.printOn = 0;

fprintf('Starting...');


%% Loopy BP
par.relchg = 0.01;
par.maxNumIter = 100;
tic;
[mu_R,mu_L,mu_U,mu_D] = lbp(Jv, Jh, H, par);
toc;

% Compute marginals
pX = mu_R.*mu_L.*mu_U.*mu_D.*exp(cat(3,-H,H));
pX = bsxfun(@rdivide, pX, sum(pX,3));


%% SMC-PGM
lZ1 = zeros(1,MC);
for(j = 1:MC)
    tic;
    [X1,W1,lZ1(j), ess_log1, Alog1] = smcpgm(Jv, Jh, H, N, par);
    toc;
end
ind1 = resampling(W1,par.resampling);
X1hat = sum(bsxfun(@times, X1, reshape(W1,[1 1 N])),3);
V1hat = sum(bsxfun(@times, bsxfun(@minus, X1, X1hat).^2, reshape(W1,[1 1 N])),3);

%% SMC-PGM-Twisted
lZ2 = zeros(1,MC);
for(j = 1:MC)
    tic;
    [X2,W2,lZ2(j), ess_log2, Alog2] = smcpgmtwist(Jv, Jh, H, mu_R, mu_L, mu_U, mu_D, N, par);
    toc;
end

X2hat = sum(bsxfun(@times, X2, reshape(W2,[1 1 N])),3);
V2hat = sum(bsxfun(@times, bsxfun(@minus, X2, X1hat).^2, reshape(W2,[1 1 N])),3);

%% ASIR
tic;
[X5,W5,lZ5, alpha_log, ess_log] = smcsampler(Jv, Jh, H, N, linspace(0,1,1000), par);
ind5 = resampling(W5,par.resampling);
toc;
X5hat = sum(bsxfun(@times, X5, reshape(W5,[1 1 N])),3);
V5hat = sum(bsxfun(@times, bsxfun(@minus, X5, X5hat).^2, reshape(W5,[1 1 N])),3);