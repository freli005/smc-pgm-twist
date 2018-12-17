% Test inference methods on a very small model where we can compute the
% exact likelihood using brute force.

clear
addpath('./helpers');
addpath('./methods');

%% Generate (or load) data
file = 'sanity';
load(file);
% 
% K = 4; % Number of topics
% V = 9294; % Size of vocabulary (each word is represented by an integer in 1:V)
% T = 10; % Number of words in document 
% 
% % Word distributions (corresponds to \Phi^T in the paper)
% topics = rand(K, V); 
% topics = bsxfun(@rdivide, topics, sum(topics, 2));
% 
% % Topic distribution prior: concentration parameters of Dirichlet
% % distribution from which document topics are drawn (corresponds to \alpha
% % in the paper)
% topic_prior = rand(K, 1); 
% topic_prior = 1 * topic_prior / sum(topic_prior);
% 
% % Random words in document
% words = ceil(rand(1, T) * V);


%% Compute exact likelihood (code by Murray, 2009)
exact = ldae_dumb_exact(words,topics,topic_prior);

%% Run EP baseline (Minka & Laffery, 2002) 
tic;
[lZep1,beta1] = ep_minka_lafferty(words, topics, topic_prior,5000);  % One factor per word in vocab
g1 = bsxfun(@plus, topic_prior, cumsum(beta1,2,'reverse'));
toc;

tic;
[lZep2,beta2] = ep_minka_lafferty_reverse(words, topics, topic_prior,5000); % One factor per word in doc
g2 = bsxfun(@plus, topic_prior, cumsum(beta2,2,'reverse'));
toc;

abs(lZep1-exact)/abs(lZep2-exact)

%% Test SMC methods
indepIter = 100; % Run each method #indepIter times 

% Varying number of particles (from 1 to 1000)
numN = 200;
Nvec = unique( round(logspace(0,3,numN)));
numN = length(Nvec);

lZsmc = zeros(numN,indepIter); % SMC-Base
lZtwist = zeros(numN,indepIter); % SMC-Twist

for iIter = 1:indepIter
    for nIter = 1:length(Nvec)
        nrParticles = Nvec(nIter);
        
        % SMC-Base
        tic;
        lZsmc(nIter,iIter) = fapf(words,topics,topic_prior,nrParticles);
        toc;
        
        % SMC-Twist
        tic;
        lZtwist(nIter,iIter) = fapf_twist(words,topics,topic_prior,g2,nrParticles);
        toc;
    end
end

%% Plot

mse_base = mean( (lZsmc - exact).^2, 2);
mse_twist = mean( (lZtwist - exact).^2, 2);
se_ep = (lZep2-exact)^2;

figure(1)
loglog(Nvec([1 end]), se_ep*[1 1],'k:'); hold on;
loglog(Nvec, mse_base,'r');
loglog(Nvec, mse_twist,'b');
legend('EP baseline','SMC-Base','SMC-Twist');

%% save
%save ./results/sanity-results.mat