%clear;file='newsgroups';test_real_multidoc;save ./results/newsgroup-epm3;clear;file='medline';test_real_multidoc;save ./results/medline-epm3;

addpath('./helpers');
addpath('./methods');

%% Test methods
numDocs = 10;   % Use the first #numDocs documents
indepIter = 50; % Run each method #indepIter times 
numN = 3;
Nvec = [50 250 1000];
load(file);

lZsmc = zeros(numN,numDocs,indepIter);
lZtwist = zeros(numN,numDocs,indepIter);
lZtwist2 = zeros(numN,numDocs,indepIter);
lZep = zeros(numDocs,1);

for iDoc = 1:numDocs
    iDoc
    tic;
    %[lZep(iDoc),ep_posterior] = ep2_sequential(docs{iDoc}, topics, topic_prior); % One factor per word in doc
    [lZep(iDoc),beta] = ep_minka3(docs{iDoc}, topics, topic_prior,5000); % One factor per word in doc
    ep_posterior = bsxfun(@plus, topic_prior, cumsum(beta,2,'reverse'));
    toc;
    toc;

    for(iN = 1:numN)
        nrParticles = Nvec(iN);
        for iIter = 1:indepIter
            tic;
            lZsmc(iN,iDoc,iIter) = fapf(docs{iDoc},topics,topic_prior,nrParticles);
            toc;

            tic;
            lZtwist(iN,iDoc,iIter) = fapf_twist(docs{iDoc},topics,topic_prior,ep_posterior,nrParticles);
            toc;

            %tic;
            %lZtwist2(iN,iDoc,iIter) = fapf_twist_reg(docs{iDoc},topics,topic_prior,ep_posterior,0.05,nrParticles);
            %toc;
        end
    end
end

%% Plot
% reestimate likelihood with "pure" EP
lZep0 = zeros(numDocs,1);
for iDoc=1:numDocs
    lZep0(iDoc) = ep_minka(docs{iDoc}, topics, topic_prior,5000); % One factor per word in doc
end

%%
% figure;
% for(iDoc = 1:numDocs)
%     h(1) = plot(iDoc + [-0.1 0.4],lZep(iDoc)*[1,1],'r--'); hold on;
%     h(2) = plot(iDoc+0.05*randn(1,indepIter),lZsmc(iDoc,:),'b.');
%     h(3) = plot(iDoc+0.2+0.05*randn(1,indepIter),lZtwist(iDoc,:),'g.');
%     h(4) = plot(iDoc+0.3+0.05*randn(1,indepIter),lZtwist2(iDoc,:),'r.');
% end
% legend(h,'EP','SMC','Twist','Twist+reg');
    
figure;
h = [];
h(1) = plot([0 numN+1],sum(lZep0)*[1,1],'r--'); hold on;

for(nIter = 1:numN)
    h(2) = plot(nIter+0.05*randn(1,indepIter),squeeze(sum(lZsmc(nIter,:,:),2)),'b.');
    h(3) = plot(nIter+0.2+0.05*randn(1,indepIter),squeeze(sum(lZtwist(nIter,:,:),2)),'g.');
    %h(4) = plot(nIter+0.4+0.05*randn(1,indepIter),squeeze(sum(lZtwist2(nIter,:,:),2)),'r.');
end
legend(h,'EP','SMC','Twist','Twist+reg');
set(gca,'XTick',1:numN,'XTickLabel',Nvec)


% figure;
% h(1) = plot([-0.1 0.4],sum( lZep )*[1,1],'r--'); hold on;
% h(2) = plot(0.05*randn(1,indepIter),sum(lZsmc,1),'b.');
% h(3) = plot(0.2+0.05*randn(1,indepIter),sum(lZtwist,1),'g.');
% h(4) = plot(0.3+0.05*randn(1,indepIter),sum(lZtwist2,1),'r.');
% legend(h,'EP','SMC','Twist','Twist+reg');

%%
load newsgroup-epm3.mat
lZbase = lZsmc;
lZep = lZep0;
save lda-newsgroup lZbase lZtwist lZep Nvec
