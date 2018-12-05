function [mu_R,mu_L,mu_U,mu_D,lZ] = lbp(Jv, Jh, H, par)
% Loopy belief propagation for Ising model

par.method = 'lbp';

% For free boundary conditions, assume periodic with zero interaction
if(par.periodic)
    [nx,ny] = size(Jv); % Size of edge set (same for Jh)
    if(size(Jh,1)~=nx || size(Jh,2)~=ny)
        error('Wrong size of edge potentials!')
    end
else
    nx = size(Jv,1)+1;
    ny = size(Jh,2)+1;
    
    if(size(Jh,1)~=nx || size(Jv,2)~=ny)
        error('Wrong size of edge potentials!')
    end    

    Jv = [Jv ; zeros(1,ny)];
    Jh = [Jh zeros(nx,1)];
end

% Set up messages. Note that there are nx*ny nodes and 2*nx*ny edges in the
% MRF. This means that we have 2*nx*ny factors in the corresponding factor
% graph and 4*nx*ny edges in this factor graph. Each edge is furthermore
% associated with 2 messages (variable->factor + factor->variable), which
% is why we have in total 8 message arrays here (verbose implementation!)

% Indices for messages are such that lambda_<..>(iX,iY,:) depend on
% variable X(iX,iY) (and similarly for mu)

% Variable -> factor messages; the "2" is since each x_k \in {-1,+1} (perhaps redundant)
lambda_R = ones(nx,ny,2); % Horizontal, going right
lambda_L = ones(nx,ny,2); % Horizontal, going left
lambda_D = ones(nx,ny,2); % Vertical, going down
lambda_U = ones(nx,ny,2); % Vertical, going up
% Factor -> variable messages; the "2" is since each x_k \in {-1,+1} (perhaps redundant)
%
% This is quite fragile, but setting ones here should initialize the
% var->factor messages as one in first iteration
mu_R = ones(nx,ny,2); % Horizontal, going right
mu_L = ones(nx,ny,2); % Horizontal, going left
mu_D = ones(nx,ny,2); % Vertical, going down
mu_U = ones(nx,ny,2); % Vertical, going up

% The "messages" from the local factors are constant and simply given by these factors
mu_H = cat(3,exp(-H),exp(H));

if(par.relchg) % Store all old messages to be able to compute relative change
    msg_old = cat(1,lambda_R(:), lambda_L(:), lambda_D(:), lambda_U(:), mu_R(:), mu_L(:), mu_D(:), mu_U(:));
end


converged = false;
numIters = 0;
while(~converged)
    % Each iteration of the LBP loop, we loop over all sites twice. In the
    % first round we send messages right+down, in the second pass we move
    % backward and send messages left+up. This should correspond roughly to
    % a "forward-backward sweep" (in particular, if the model is a chain it
    % should converge in one iteration).
            
    % Top left to bottom right, var->fact followed by fact->var
    for iX = 1:nx % Loop over rows
        for iY = 1:ny % Loop over cols
            % Variable -> factor messages, right+down; Note indexing
            % convention here, that all incoming factor messages depend on
            % variable X(iX,iY) and hence have the same indices
            lambda_R(iX,iY,:) = mu_R(iX,iY,:).*mu_D(iX,iY,:).*mu_U(iX,iY,:).*mu_H(iX,iY,:);
            lambda_D(iX,iY,:) = mu_R(iX,iY,:).*mu_D(iX,iY,:).*mu_L(iX,iY,:).*mu_H(iX,iY,:);
            
            % Normalize?
            lambda_R(iX,iY,:) = lambda_R(iX,iY,:)/sum(lambda_R(iX,iY,:));
            lambda_D(iX,iY,:) = lambda_D(iX,iY,:)/sum(lambda_D(iX,iY,:));
            
            % Factor -> variable messages, right+down; Here we update the
            % messages along edges for which X(iX,iY) is "responsible",
            % which means that the messages depend on the neighboring
            % variables
            
            % First find indices
            iD = mod(iX,nx)+1;
            %iU = mod(iX-2,nx)+1;
            iR = mod(iY,ny)+1;
            %iL = mod(iY-2,ny)+1;
            
            % We sum over X(iX,iY) \in {-1,1} for X(iR,iY) = [-1,1], and
            % similarly for X(iX,iD) = [-1,1]
            mu_R(iX,iR,:) = lambda_R(iX,iY,1)*exp(Jh(iX,iY)*[1;-1]) + lambda_R(iX,iY,2)*exp(Jh(iX,iY)*[-1;1]);
            mu_D(iD,iY,:) = lambda_D(iX,iY,1)*exp(Jv(iX,iY)*[1;-1]) + lambda_D(iX,iY,2)*exp(Jv(iX,iY)*[-1;1]);
            
            % Normalize?
            mu_R(iX,iR,:) = mu_R(iX,iR,:)/sum(mu_R(iX,iR,:));
            mu_D(iD,iY,:) = mu_D(iD,iY,:)/sum(mu_D(iD,iY,:));            
        end
    end
    
    % Bottom right to top left, var->fact followed by fact->var
    for iX = nx:-1:1 % Loop over rows
        for iY = ny:-1:1 % Loop over cols
            % Variable -> factor messages, left+up; Messages propagate
            % "towards" X(iX,iY), hence the shift in indexing here

            % First find indices
            iD = mod(iX,nx)+1;
            %iU = mod(iX-2,nx)+1;
            iR = mod(iY,ny)+1;
            %iL = mod(iY-2,ny)+1;
            
            lambda_L(iX,iR,:) = mu_L(iX,iR,:).*mu_D(iX,iR,:).*mu_U(iX,iR,:).*mu_H(iX,iR,:);
            lambda_U(iD,iY,:) = mu_U(iD,iY,:).*mu_L(iD,iY,:).*mu_R(iD,iY,:).*mu_H(iD,iY,:);
            
            lambda_L(iX,iR,:) = lambda_L(iX,iR,:)/sum(lambda_L(iX,iR,:));
            lambda_U(iD,iY,:) = lambda_U(iD,iY,:)/sum(lambda_U(iD,iY,:));
            
            
            % Factor -> variable messages, left+up;
            mu_L(iX,iY,:) = lambda_L(iX,iR,1)*exp(Jh(iX,iY)*[1;-1]) + lambda_L(iX,iR,2)*exp(Jh(iX,iY)*[-1;1]);
            mu_U(iX,iY,:) = lambda_U(iD,iY,1)*exp(Jv(iX,iY)*[1;-1]) + lambda_U(iD,iY,2)*exp(Jv(iX,iY)*[-1;1]);

            mu_L(iX,iY,:) = mu_L(iX,iY,:)/sum(mu_L(iX,iY,:));
            mu_U(iX,iY,:) = mu_U(iX,iY,:)/sum(mu_U(iX,iY,:));
        
        end
    end
    

    if(par.relchg) % Store all old messages to be able to compute relative change
        msg_now = cat(1,lambda_R(:), lambda_L(:), lambda_D(:), lambda_U(:), mu_R(:), mu_L(:), mu_D(:), mu_U(:));
        relchg = max(abs(msg_now-msg_old)./abs(msg_old));
        msg_old = msg_now;
        converged = relchg < par.relchg;
        if(converged)
            numIters
        end
    end
    
    numIters = numIters+1;
    converged = converged || numIters >= par.maxNumIter;
end