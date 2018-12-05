addpath('./helpers');
addpath('./methods');
clear

%% Generate or load data
% with any method worth considering.
%file = 'sanity2';
%load(file);

T = 4;
V = 22294;
n = 10;
topics = rand(T, V);
topics = bsxfun(@rdivide, topics, sum(topics, 2));
topic_prior = rand(T, 1);
topic_prior = 0.001 * topic_prior / sum(topic_prior);
words = ceil(rand(1, n) * V);

exact = ldae_dumb_exact(words,topics,topic_prior);

tic;
[lZep1,beta1] = ep_minka(words(2:end), topics, topic_prior,5000);  % One factor per word in vocab
g1 = bsxfun(@plus, topic_prior, cumsum(beta1,2,'reverse'));
toc;

tic;
[lZep2,beta2] = ep2(words, topics, topic_prior,5000);  % One factor per word in vocab
g2 = bsxfun(@plus, topic_prior, cumsum(beta2,2,'reverse'));
toc;

tic;
[lZep3,g3] = ep_minka_sequential(words, topics, topic_prior,5000); % One factor per word in doc
%[lZep3,beta3] = ep_minka3(words, topics, topic_prior,5000); % One factor per word in doc
%g3 = bsxfun(@plus, topic_prior, cumsum(beta3,2,'reverse'));
toc;

% tt = 0.9;
% g3 = bsxfun(@plus, tt*topic_prior, (1-tt)*g3);


abs(lZep1-exact)/abs(lZep3-exact)


%% Test methods
indepIter = 20; % Run each method #indepIter times 
numN = 3;
Nvec = [50 250 1000];

lZsmc = zeros(numN,indepIter);
lZtwist = zeros(numN,indepIter);
lZtwist2 = zeros(numN,indepIter);

for iIter = 1:indepIter
    for nIter = 1:length(Nvec)
        nrParticles = Nvec(nIter);
        
        % SMC
        tic;
        lZsmc(nIter,iIter) = fapf(words,topics,topic_prior,nrParticles);
        toc;
        
        tic;
        lZtwist(nIter,iIter) = fapf_twist(words,topics,topic_prior,g3,nrParticles);
        toc;

        tic;
        lZtwist2(nIter,iIter) = fapf_twist_reg(words,topics,topic_prior,g3,0.001,nrParticles);
        toc;
        
    end
end

%% Plot
figure;
h(1) = plot([0 numN+1],lZep3*[1,1],'r--'); hold on;
h(5) = plot([0 numN+1],exact*[1,1],'k--'); hold on;

for(nIter = 1:numN)
    h(2) = plot(nIter+0.05*randn(1,indepIter),lZsmc(nIter,:),'b.');
    h(3) = plot(nIter+0.2+0.05*randn(1,indepIter),lZtwist(nIter,:),'g.');
    h(4) = plot(nIter+0.4+0.05*randn(1,indepIter),lZtwist2(nIter,:),'r.');
end
legend(h,'EP','SMC','Twist','Twist+reg','Exact');
set(gca,'XTick',1:numN,'XTickLabel',Nvec)
