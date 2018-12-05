addpath('./helpers');
addpath('./methods');

%% Generate or load data
% with any method worth considering.
%file = 'sanity2';
%load(file);

T = 50;
V = 22294;
n = 100;
topics = rand(T, V);
topics = bsxfun(@rdivide, topics, sum(topics, 2));
topic_prior = rand(T, 1);
%load medline topic_prior %??
%topic_prior=topic_prior(1:T);

topic_prior = 2 * topic_prior / sum(topic_prior);

words = ceil(rand(1, n) * V);


tic;
[lZep1,beta1] = ep_minka(words, topics, topic_prior, 5000);  % One factor per word in vocab
toc;

% tic;
%[lZep2,beta2] = ep2(words, topics, topic_prior, 5000);  % One factor per word in vocab
% ep_posterior = bsxfun(@plus, topic_prior, cumsum(beta2(:,end:-1:1),2));
% ep_posterior = ep_posterior(:,end:-1:1);
% % "Temper"
% tmp1 = cumsum(beta2(:,end:-1:1),2);
% tmp1 = tmp1(:,end:-1:1);
% AA = bsxfun(@rdivide, 0.99*topic_prior, -tmp1);
% alpha = min( AA(AA >=0) );
% ep_posterior = bsxfun(@plus, topic_prior, alpha*tmp1);
% toc;
% tic;
% [lZep,ep_posterior] = ep2_sequential(words, topics, topic_prior, 50); % One factor per word in doc
% toc;
tic;
[lZep3,beta3] = ep_minka3(words, topics, topic_prior,5000); % One factor per word in doc
g3 = bsxfun(@plus, topic_prior, cumsum(beta3,2,'reverse'));
toc;
% 
% tic;
% [lZep4,g4] = ep_minka_sequential(words, topics, topic_prior,5000); % One factor per word in doc
% toc;
% 
% 
% tic;
% for(j=1:n)
%     [lZep5,beta5] = ep_minka(words(j:end), topics, topic_prior, 5000);  % One factor per word in vocab
%     g5(:,j) = topic_prior + sum(beta5,2);
% end
% toc;
ep_posterior = g3;

%%

% for(j=1:n)
%    figure(1); plot(g3(:,j))
%     hold all; plot(g4(:,j))
%     plot(g5(:,j)) 
%     hold off;
%     legend('m3','seq','brute');
%     pause;
%     
% end
%% Test methods
indepIter = 10; % Run each method #indepIter times 
numN = 2;
Nvec = [50 1000];

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

        %tic;
        %lZtwist2(nIter,iIter) = fapf_twist_reg(words,topics,topic_prior,ep_posterior,0.05,nrParticles);
        %lZtwist2(nIter,iIter) = fapf_twist(words,topics,topic_prior,ep_posterior,nrParticles);
        %toc;
        
    end
end

%% Plot
figure;
h = [];
h(1) = plot([0 numN+1],lZep3*[1,1],'r--'); hold on;

for(nIter = 1:numN)
    h(2) = plot(nIter+0.05*randn(1,indepIter),lZsmc(nIter,:),'b.');
    h(3) = plot(nIter+0.2+0.05*randn(1,indepIter),lZtwist(nIter,:),'g.');
    %h(4) = plot(nIter+0.4+0.05*randn(1,indepIter),lZtwist2(nIter,:),'r.');
end
legend(h,'EP','SMC','Twist','Twist+reg');
set(gca,'XTick',1:numN,'XTickLabel',Nvec)

