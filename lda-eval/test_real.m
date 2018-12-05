addpath('./helpers');
addpath('./methods');

%% Load data
file = 'newsgroups';
load(file);
words = docs{1};

tic;
[lZep2,beta2] = ep_minka(words, topics, topic_prior, 5000);  % One factor per word in vocab
tmp1 = cumsum(beta2(:,end:-1:1),2);
tmp1 = tmp1(:,end:-1:1);
AA = bsxfun(@rdivide, 0.99*topic_prior, -tmp1);
%alpha = min( AA(AA >=0) );
AA(AA<=0) = 1;
AA(AA>1) = 1;
alpha = min(AA,[],1)
g2 = bsxfun(@plus, topic_prior, bsxfun(@times, alpha, tmp1));
toc;

% tic;
% [lZep,ep_posterior] = ep2_sequential(words, topics, topic_prior, 50); % One factor per word in doc
% toc;
tic;
[lZep3,beta3] = ep_minka3(words, topics, topic_prior); % One factor per word in doc
g3 = bsxfun(@plus, topic_prior, cumsum(beta3,2,'reverse'));
toc;

ep_posterior = g3;

%% Test methods
indepIter = 10; % Run each method #indepIter times 
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
        lZtwist(nIter,iIter) = fapf_twist(words,topics,topic_prior,ep_posterior,nrParticles);
        toc;

        tic;
%         lZtwist2(nIter,iIter) = fapf_twist_reg(words,topics,topic_prior,g3,0.05,nrParticles);
        lZtwist2(nIter,iIter) = fapf_twist(words,topics,topic_prior,g2,nrParticles);
        toc;
        
    end
end

%% Plot
figure;
h(1) = plot([0 numN+1],lZep3*[1,1],'r--'); hold on;

for(nIter = 1:numN)
    h(2) = plot(nIter+0.05*randn(1,indepIter),lZsmc(nIter,:),'b.');
    h(3) = plot(nIter+0.2+0.05*randn(1,indepIter),lZtwist(nIter,:),'g.');
    h(4) = plot(nIter+0.4+0.05*randn(1,indepIter),lZtwist2(nIter,:),'r.');
end
legend(h,'EP','SMC','Twist','Twist+reg');
set(gca,'XTick',1:numN,'XTickLabel',Nvec)

