addpath('../methods')
addpath('../utils')

dataset = 1;
load(sprintf('res_%i_N64',dataset)); % Load model

par.printOn = 1;
[Xgt,Wgt,lZgt, alpha_log, ess_log] = smcsampler(Jv, Jh, H, 1024, linspace(0,1,10000), par);

