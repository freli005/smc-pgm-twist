% Test inference methods on real data (single document)

clear
addpath('./helpers');
addpath('./methods');

%% Load data
file = 'newsgroups'; % or 'medline'
load(file);
words = docs{1};

tic;
[lZep1,beta1] = ep_minka_lafferty(words, topics, topic_prior, 5000);  % One factor per word in vocab
% tmp1 = cumsum(beta1(:,end:-1:1),2);
% tmp1 = tmp1(:,end:-1:1);
% AA = bsxfun(@rdivide, 0.99*topic_prior, -tmp1);
% %alpha = min( AA(AA >=0) );
% AA(AA<=0) = 1;
% AA(AA>1) = 1;
% alpha = min(AA,[],1)
% g2 = bsxfun(@plus, topic_prior, bsxfun(@times, alpha, tmp1));
toc;

tic;
[lZep2,beta2] = ep_minka_lafferty_reverse(words, topics, topic_prior);
ep_posterior = bsxfun(@plus, topic_prior, cumsum(beta2,2,'reverse'));
toc;

lZep2-lZep1

%% Test methods
indepIter = 10; % Run each method #indepIter times 
numN = 3;
Nvec = [50 250 1000];

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


