addpath('./helpers');
addpath('./methods');


%% Generate or load data
file = 'sanity2';
load(file);
% 
% T = 4;
% V = 9294;
% n = 10;
% topics = rand(T, V);
% topics = bsxfun(@rdivide, topics, sum(topics, 2));
% topic_prior = rand(T, 1);
% topic_prior = 2 * topic_prior / sum(topic_prior);
% words = ceil(rand(1, n) * V);

exact = ldae_dumb_exact(words,topics,topic_prior);

tic;
[lZep2,beta2] = ep2(words, topics, topic_prior,5000);  % One factor per word in vocab
toc;
tic;
%[lZep3,g3] = ep2_sequential(words, topics, topic_prior); % One factor per word in doc
[lZep3,beta3] = ep_minka3(words, topics, topic_prior,5000); % One factor per word in doc
g3 = bsxfun(@plus, topic_prior, cumsum(beta3,2,'reverse'));
toc;

abs(lZep2-exact)/abs(lZep3-exact)

% [lZep,beta] = ep2(words, topics, topic_prior); % One factor per word in doc

%% Test methods
indepIter = 100; % Run each method #indepIter times 
numN = 200;
%nrParticles = 25; % Multiplied by length(words) for the SMC sampler
Nvec = unique( round(logspace(0,3,numN)));
numN = length(Nvec);

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
        
%         tic;
%         lZtwist2(nIter,iIter) = fapf_twist_reg(words,topics,topic_prior,g3,0.05,nrParticles);
%         toc;
        
        % LRS
        %tic;
        %lZlrs(iIter) = lrs2(words,topics,topic_prior,nrParticles);
        %toc;
    end
end

%% Plot
%plot([1 indepIter],exact*[1,1],'k--'); hold on;
%plot([1 indepIter],lZep*[1,1],'r--');
figure;
h = [];
plot(Nvec([1,end]),exact*[1,1],'k--'); hold on;
plot(Nvec([1,end]),lZep3*[1,1],'r--');

plot(Nvec,lZsmc,'b-');
plot(Nvec,lZtwist,'g-');
%plot(Nvec,lZtwist2,'r-');
legend(h,'Exact','EP','SMC','Twist','Twist+reg');


%%
mse_base = mean( (lZsmc - exact).^2, 2);
mse_twist = mean( (lZtwist - exact).^2, 2);
se_ep = (lZep2-exact)^2;

8
print(1,'-dpdf','lda-sanity')

%% save
save ./results/sanity-epm3.mat
save ./results/lda-sanity.mat mse_base mse_twist se_ep Nvec
