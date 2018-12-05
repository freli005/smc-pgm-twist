addpath('./utils');
addpath('./methods');

filename = './results/res_%i_N%i'

%%%% MODEL SPECIFICATION

% Fixed parameters
par.periodic = true;
M = 2^4; nx = M; ny = M;

% Varying parameters
%vec_invtemp = [0.5 1 2];

%vec_invtemp = [0.9 1 1.1]/2.269185; % Centered at critical temperature
%H = 2*rand(nx,ny)-1; % External field
%num_models = length(vec_invtemp);

%load from prev run
load('./results/gt_1.mat','H','vec_invtemp')
num_models = 3; % Only run for first data set for now


%%%% Algorithmic settings

% Varying parameters
%vec_N = [100 500 1000];
vec_N = 2.^(6:2:11);
num_N = length(vec_N);
num_MC = 50;

% Static parameters
par.resampling = 3;
par.ESS_threshold = 0.5;
warning off;

par.CESS_threshold = 0.995;
par.CESS_tolerance = 1e-5;
par.plotOn = 0;
par.printOn = 0;

%%%% Data storage parameters
num_methods = 2; % SMC-PGM, SMC-Twist
num_sites = 10; % Number of sites to store all particles and weights for

%%%% Loop over models
for(j = 1:num_models)
    par.dataset = j;
    % Load data
    inv_temp = vec_invtemp(j);
  
    %%%% Precomputation of model quantities
    if(par.periodic)
        Jv = inv_temp*ones(M);
        Jh = inv_temp*ones(M);
    else
        Jv = inv_temp*ones(M-1,M);
        Jh = inv_temp*ones(M,M-1);
    end
    
    % Find num_sites sites to monitor
    INDS = round(linspace(1,M^2,num_sites));
    %[inds_x, inds_y] = ind2sub([M M],INDS);
    
    %%%% Loop over settings (#particles)
    for(n = 1:num_N)
        N = vec_N(n);
        
        %%%% Allocate memory
        XX = zeros(num_sites, N, num_methods, num_MC); % All particles
        WW = zeros(N, num_methods, num_MC); % All weights
        lZ = zeros(num_methods, num_MC);  % log-normalising-constant
        cpu_times = zeros(num_methods, num_MC);
        
        Xhat = zeros(nx, ny, num_methods, num_MC);  % E[X]
        X2hat = zeros(nx, ny, num_methods, num_MC); % E[X^2]
        X4hat = zeros(nx, ny, num_methods, num_MC); % E[X^4]
        PrbPos = zeros(nx, ny, num_methods, num_MC);% P(X > 0)
        % Auxiliary estimates:
        % 1-2: Energy, E[(E,E^2)]
        % 3-4: Magnetization, E[(M,M^2)]
        AUXhat = zeros(4,num_methods,num_MC);
        
        for(mc = 1:num_MC) % Number of runs           
            % Data storage for this worker
            XX_tmp = zeros(num_sites, N, num_methods); % All particles
            WW_tmp = zeros(N, num_methods); % All weights
            lZ_tmp = zeros(num_methods,1);  % log-normalising-constant
            cpu_times_tmp = zeros(num_methods,1);
            
            Xhat_tmp = zeros(nx, ny, num_methods);  % E[X]
            X2hat_tmp = zeros(nx, ny, num_methods); % E[X^2]
            X4hat_tmp = zeros(nx, ny, num_methods); % E[X^4]
            PrbPos_tmp = zeros(nx, ny, num_methods);% P(X > 0)
            AUXhat_tmp = zeros(4,num_methods);
            
            for(method = 1:num_methods)
                fprintf('Starting, data=%i, N=%i, method=%i (%i)...\n',j,N,method,mc);

                if(method == 1) %%% SMC-PGM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    tic;
                    [partX, W, lZ_tmp(method), ess_log] = smcpgm(Jv, Jh, H, N, par);
                    cpu_times_tmp(method) = toc;
                elseif(method==2) %%% SMC-Twist %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    tic;
                    [mu_R,mu_L,mu_U,mu_D] = lbp(Jv, Jh, H, par);
                    [partX, W, lZ_tmp(method), ess_log] = smcpgmtwist(Jv, Jh, H, mu_R, mu_L, mu_U, mu_D, N, par);
                    cpu_times_tmp(method) = toc;                    
                end
                
                fprintf('Elapsed time is %f seconds.\n',cpu_times_tmp(method));
                
                % Compute outputs
                tmp = reshape(partX,[M^2 N]);
                XX_tmp(:,:,method) = tmp(INDS,:);
                WW_tmp(:,method) = W;
                % Estimates
                Xhat_tmp(:,:,method) = sum(bsxfun(@times, partX, reshape(W,[1 1 N])),3);
                X2hat_tmp(:,:,method) = sum(bsxfun(@times, partX.^2, reshape(W,[1 1 N])),3);
                X4hat_tmp(:,:,method) = sum(bsxfun(@times, partX.^4, reshape(W,[1 1 N])),3);
                PrbPos_tmp(:,:,method) = sum(bsxfun(@times, partX > 0, reshape(W,[1 1 N])),3);
                % Auxiliary estimators
                engy = -sum_diff_x(partX,Jv,Jh,H); % Energy
                AUXhat_tmp(1,method) = (engy')*W(:);
                AUXhat_tmp(2,method) = ((engy.^2)')*W(:);
                %
                mag = mean(mean(partX,1),2); % Magnetisation
                mag = mag(:);
                AUXhat_tmp(3,method) = (mag')*W(:);
                AUXhat_tmp(4,method) = ((mag.^2)')*W(:);                
            end        
        
            % Store data
            XX(:,:,:,mc) = XX_tmp;
            WW(:,:,mc) = WW_tmp;
            lZ(:,mc) = lZ_tmp;
            cpu_times(:,mc) = cpu_times_tmp;
            
            Xhat(:,:,:,mc) = Xhat_tmp;
            X2hat(:,:,:,mc) = X2hat_tmp;
            X4hat(:,:,:,mc) = X4hat_tmp;
            PrbPos(:,:,:,mc) = PrbPos_tmp;
            
            AUXhat(:,:,mc) = AUXhat_tmp;
            
        end
        
        % Save data
        save(sprintf(filename, j, N));
    end
end



