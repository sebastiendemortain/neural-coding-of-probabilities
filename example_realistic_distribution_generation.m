clear all; close all; clc;

n_subject = 1;
n_block = 1;

% Sequence size
L = 380;

s = cell(n_subject, n_block);
gen_prob = cell(n_subject, n_block);
out_io = cell(n_subject, n_block);

% Ideal observer for specific sequences
p = genpath('C:\Users\Sébastien\Documents\DTU\Master thesis\code');
addpath(p);

for i_subject=1:n_subject
    for i_block=1:n_block
        [s_tmp, gen_prob_tmp] = generate_sequence(L);

        s{i_subject, i_block} = s_tmp;
        gen_prob{i_subject, i_block} = gen_prob_tmp;

        %% EXAMPLE 1: COMPUTE THE IDEAL OBSERVER WITH JUMPS (HMM)
        %  ======================================================

        pJump = 1/75; % Value by default

        % Set parameters
        in.s            = s_tmp;                % sequence
        in.learned      = 'transition';     % estimate transition
        in.jump         = 1;                % estimate with jumps
        in.mode         = 'HMM';            % use the HMM (not sampling) algorithm
        in.opt.pJ       = pJump;            % a priori probability that a jump occur at each outcome
        n               = 50;               % resolution of the univariate probability grid
        in.opt.pgrid    = linspace(0,1,n);  % estimation probability grid
        in.opt.Alpha0   = ones(n)/(n^2);    % uniform prior on transition probabilities
        in.verbose      = 1;                % to check that no default values are used.

        % Compute the observer
        out_io{i_subject, i_block} = IdealObserver(in);
    end
end

% Save the data
savefile = strcat('data/simu/ideal_observer_',num2str(n_subject),'subjects_',num2str(n_block),'blocks_',num2str(L),'stimuli_',in.mode,'.mat');
save(savefile, 'out_io', 's', 'gen_prob');
