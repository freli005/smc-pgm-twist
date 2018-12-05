% Stripped down plotting for the Ising model
% - Variance of log-NC vs. CPU Time
% - Variance of sum_ij E[X_ij]/M^2 vs. CPU Time
% - Variance of E[energy] vs. CPU Time
% 
% Also generates tabular values.

close all;
clear;
dataset = 1;
extra = 0;
printON = 0;
zoomIn = 0;


%% Load data and compute estimates
vec_N = 2.^(6:2:11);
numN = length(vec_N);
M = 16;
T = log2(M);
num_methods = 2; % Excluding Gibbs
num_MC = 50;

num_est = 3; % log-NC, Xhat, Ehat
EstToPlotSMC = zeros(num_MC,num_methods, numN, num_est); % Estimator variances to plot
CPU_SMC = zeros(num_methods,numN);
%EstToPlotGibbs = zeros(num_MC, 1, numGibbs, num_est); % Estimator variances to plot

% SMC samplers
for(ii = 1:numN)
    N = vec_N(ii);
    load(sprintf('res_%i_N%i',dataset,N),'lZ','Xhat','AUXhat','cpu_times');
    %CPU_SMC(:,ii) = median(cpu_times,2);
    CPU_SMC(:,ii) = min(cpu_times,[],2);
        
    % log-NC
    ev = lZ;
    EstToPlotSMC(:,:,ii,1) = ev';

    % Xhat
    ev = squeeze( mean(mean(Xhat.^2,1),2) );
    EstToPlotSMC(:,:,ii,2) = ev';
    
    % Ehat
    ev = squeeze(AUXhat(1,:,:));
    EstToPlotSMC(:,:,ii,3) = ev';
end

% % Gibbs sampler
% tmp = vec_N;
% load(sprintf('gibbs_%i',dataset),'vec_N','Xhat','AUXhat','cpu_times');
% vec_MCMC = vec_N;
% vec_N = tmp;
% if(burnin > 0)
%     burnMCMC = vec_MCMC(burnin);
% else
%     burnMCMC = 0;
% end
% numMCMC_tmp = vec_MCMC(burnin+1:end)'; % Number of iterations up to each estimate
% numMCMC_inbtw = [vec_MCMC(1) diff(vec_MCMC)]; % Number of iterations in between estimates
% numMCMC_inbtw = numMCMC_inbtw(burnin+1:end);
% cpu_times_tmp = min(cumsum(cpu_times,1),[],2);
% CPU_Gibbs = cpu_times_tmp(burnin+1:end);
% 
% % log-NC
% EstToPlotGibbs(:,1,:,1) = NaN;
% % Xhat
% Xhat_tmp = bsxfun(@rdivide, cumsum(bsxfun(@times, Xhat(:,:,burnin+1:end,:), reshape(numMCMC_inbtw,1,1,[])),3), reshape(numMCMC_tmp-burnMCMC,1,1,[]));
% ev = squeeze(mean(mean(Xhat_tmp.^2,1),2));
% EstToPlotGibbs(:,1,:,2) = ev';
% % % Ehat
% AUXhat_tmp = bsxfun(@rdivide, cumsum(bsxfun(@times, AUXhat(:,burnin+1:end,:), reshape(numMCMC_inbtw,1,[])),2), reshape(numMCMC_tmp-burnMCMC,1,[]));
% ev = squeeze(AUXhat_tmp(1,:,:));
% EstToPlotGibbs(:,1,:,3) = ev';


%% Plot
close all;
colors = {'r',0.7*[0 1 0],[0 0.5 0],'k'};
method_names = {'SMC-PGM','SMC-Twist'};
axes_font_size = 10;
legend_font_size = 8;
legend_marker_size = 11;

plot_these_methods = [1 2]; % + Gibbs!

num_methods = numel(plot_these_methods);
CPU_SMC = CPU_SMC(plot_these_methods,:);
EstToPlotSMC = EstToPlotSMC(:,plot_these_methods,:,:);
method_names = method_names([plot_these_methods end]);
colors = colors([ plot_these_methods  end]);

%% log-NC
close all;
ee = 1;
figure(ee);
% if(extra)
%     hf1 = axes('position', [0.1300    0.1100    0.7750    0.7*0.8150]); % For supplement
% else
%     hf1 = axes('position', [0.1300    0.1100    0.7750    0.35*0.8150]); % For main paper
% end

D1 = 0.2;
D2 = 0.05;

xlims = [D1-0.1 numN*D1+D2+0.1]; % Hard coded; used to set limits below
load('gt_1.mat','lZgt');
plot(xlims,lZgt*[1 1],'k--'); hold on;

for(mm = 1:num_methods)
    %if(plot_these_methods(mm))
        for(nn = 1:numN)
            tmp = EstToPlotSMC(:,mm,nn,ee);
            a = min(tmp);
            b = prctile(tmp,25);
            c = prctile(tmp,50);
            d = prctile(tmp,75);
            e = max(tmp);
            
            %semilogx(CPU_SMC(mm,nn), EstToPlotSMC(:,mm,nn,ee),'.','color',colors{mm}); hold on;
            plot( (D1*nn+(mm-1)*D2 )*[1 1], [a e],'-','linewidth',3,'color',colors{mm}); hold on;
            plot( (D1*nn+(mm-1)*D2 )*[1 1], [b d],'-','linewidth',15,'color',colors{mm});
            plot( (D1*nn+(mm-1)*D2 ), c,'.','MarkerSize',12,'color','white');
            plot( (D1*nn+(mm-1)*D2 ), c,'.','MarkerSize',5,'color','black');
            %semilogx(CPU_SMC(mm,nn), c,'.','color','w');
        end
    %end
end
%xlims = get(gca,'Xlim');
%xlims = [xlims(1)-0.1, xlims(2)+0.1];

set(gca,'fontsize',axes_font_size,'Xlim',xlims,'XTick',D1*(1:numN)+D2/2,'XTicklabel',vec_N)
%xlabel('Wall-clock time (s)');
xlabel('Number of particles');

% Set up legend
lh = zeros(1,num_methods);
for(mm = 1:num_methods)
    lh(mm) = plot(NaN,NaN,'marker','square','linestyle','none','markersize',legend_marker_size,'markerfacecolor',colors{mm},'color',colors{mm});
end
lh = legend(lh,method_names{1:end-1},'location','southeast');
set(lh,'fontsize',legend_font_size,'box','off','color','none')
%squeezelegend(lh,0.8);


if(printON)
    print(ee,'-depsc2',sprintf('%slogNC',filename));
end

% %% energy
% close all;
% ee = 3;
% figure(ee);
% if(extra)
%     hf1 = axes('position', [0.1300    0.1100    0.7750    0.7*0.8150]); % For supplement
% else
%     hf1 = axes('position', [0.1300    0.1100    0.7750    0.35*0.8150]); % For main paper
% end
% 
% for(mm = 1:num_methods)
%     for(nn = 1:numN)
%         tmp = EstToPlotSMC(:,mm,nn,ee);
%         a = min(tmp);
%         b = prctile(tmp,25);
%         c = prctile(tmp,50);
%         d = prctile(tmp,75);
%         e = max(tmp);
%         
%         %semilogx(CPU_SMC(mm,nn), EstToPlotSMC(:,mm,nn,ee),'.','color',colors{mm}); hold on;
%         semilogx(CPU_SMC(mm,nn)*[1 1], [a e],'-','linewidth',2,'color',colors{mm}); hold on;
%         semilogx(CPU_SMC(mm,nn)*[1 1], [b d],'-','linewidth',5,'color',colors{mm});
%         %semilogx(CPU_SMC(mm,nn), c,'.','color','w');
%     end
% end
% 
% % Gibbs
% for(nn = 1:numGibbs)
%     tmp = EstToPlotGibbs(:,1,nn,ee);
%     a = min(tmp);
%     b = prctile(tmp,25);
%     c = prctile(tmp,50);
%     d = prctile(tmp,75);
%     e = max(tmp);
%     
%     %semilogx(CPU_SMC(mm,nn), EstToPlotSMC(:,mm,nn,ee),'.','color',colors{end}); hold on;
%     semilogx(CPU_Gibbs(nn)*[1 1], [a e],'-','linewidth',2,'color',colors{end}); hold on;
%     semilogx(CPU_Gibbs(nn)*[1 1], [b d],'-','linewidth',5,'color',colors{end});
%     %semilogx(CPU_SMC(mm,nn), c,'.','color','w');
% end
% 
% 
% set(gca,'fontsize',axes_font_size)
% xlabel('Wall-clock time (s)');
% 
% % Set up legend
% lh = zeros(1,num_methods);
% for(mm = 1:num_methods+1)
%     lh(mm) = semilogx(NaN,NaN,'marker','square','linestyle','none','markersize',legend_marker_size,'markerfacecolor',colors{mm},'color',colors{mm});
% end
% lh = legend(lh,method_names,'location','southwest');
% set(lh,'fontsize',legend_font_size,'box','off','color','none')
% squeezelegend(lh,0.8);
% 
% 
% if(printON)
%     print(ee,'-depsc2',sprintf('%sE',filename));
% end

