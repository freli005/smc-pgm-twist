% Test inference methods on a slightly larger model (than 'sanity'), but
% still with simulated data

clear
addpath('./helpers');
addpath('./methods');

%% Generate or load data

K = 200; % Number of topics
V = 22294; % Size of vocabulary (each word is represented by an integer in 1:V)
T = 400; % Number of words in document 

% Word distributions (corresponds to \Phi^T in the paper)
topics = rand(K, V);
topics = bsxfun(@rdivide, topics, sum(topics, 2));

% Topic distribution prior: concentration parameters of Dirichlet
% distribution from which document topics are drawn (corresponds to \alpha
% in the paper)
topic_prior = rand(K, 1);
topic_prior = 2 * topic_prior / sum(topic_prior);

% Generate random words in document
theta0 = gamrnd(topic_prior,1); theta0 = theta0/sum(theta0);
x0 = randsample(K,T,true,theta0);
words = zeros(1,T);
for(j = 1:T)
    topic_j = topics(x0(j),:);
    words(j) = randsample(V,1,true,topic_j);
end

tic;
[lZep1,beta1] = ep_minka_lafferty(words, topics, topic_prior, 5000);
toc;

tic;
[lZep2,beta2] = ep_minka_lafferty_reverse(words, topics, topic_prior,5000);
ep_posterior = bsxfun(@plus, topic_prior, cumsum(beta2,2,'reverse'));
toc;

lZep2-lZep1

%% Test methods
indepIter = 10; % Run each method #indepIter times 
numN = 1;
Nvec = [50];

lZsmc = zeros(numN,indepIter);
lZtwist = zeros(numN,indepIter);

for iIter = 1:indepIter
    for nIter = 1:length(Nvec)
        nrParticles = Nvec(nIter);
        
        % SMC
        tic;
        lZsmc(nIter,iIter) = fapf(words,topics,topic_prior,nrParticles);
        toc;
        
        tic;
        lZtwist(nIter,iIter) = fapf_twist(words,topics,topic_prior,ep_posterior,nrParticles);
        toc;
    end
end

%% Plot
figure;
h = [];
h(1) = plot([0 numN+1],lZep1*[1,1],'r--'); hold on;
h(2) = plot([0 numN+1],lZep2*[1,1],'g--'); hold on;
for(nIter = 1:numN)
    h(3) = plot(nIter+0.05*randn(1,indepIter),lZsmc(nIter,:),'b.');
    h(4) = plot(nIter+0.2+0.05*randn(1,indepIter),lZtwist(nIter,:),'g.');
end
legend(h,'EP1','EP2','SMC','Twist');
set(gca,'XTick',1:numN,'XTickLabel',Nvec)

