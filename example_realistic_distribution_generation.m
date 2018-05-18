clear all; close all; clc;

n_subject = 100;
n_session = 4;

% Sequence size
L = 380;

s = cell(n_subject, n_session);
gen_prob = cell(n_subject, n_session);
out_io = cell(n_subject, n_session);

% Ideal observer for specific sequences
p = genpath('C:\Users\Sébastien\Documents\DTU\Master thesis\code');
addpath(p);

for i_subject=1:n_subject
    for i_session=1:n_session
        [s_tmp, gen_prob_tmp] = generate_sequence(L);

        s{i_subject, i_session} = s_tmp;
        gen_prob{i_subject, i_session} = gen_prob_tmp;

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
        
        io_struct = IdealObserver(in);
        p1_mean = io_struct.p1_mean;
        p1_sd = io_struct.p1_mean;
        p1_dist = zeros(n, L);
        
        p1g2_dist = io_struct.p1g2_dist;
        p2g1_dist = io_struct.p2g1_dist;
        % The first distribution is indifferently p1g2 or p2g1
        p1_dist(:, 1) = p1g2_dist(:, 1);
        % Make p1_dist
        for i = 2:L
           if s_tmp(i-1) == 1
               p1_dist(:, i) = flipud(p2g1_dist(:, i));    % p(1|1)(x) = p(2|1)(1-x)
           elseif s_tmp(i-1) == 2 
               p1_dist(:, i) = p1g2_dist(:, i);
           else
               print('Error');
           end
        end
        % Compute the observer
        struct_tmp.p1_mean = p1_mean;
        struct_tmp.p1_sd = p1_sd;
        struct_tmp.p1_dist = p1_dist;
        out_io{i_subject, i_session} = struct_tmp;
    end
end

% Save the data
savefile = strcat('data/simu/ideal_observer_',num2str(n_subject),'subjects_',num2str(n_session),'sessions_',num2str(L),'stimuli_',in.mode,'.mat');
save(savefile, 'out_io');
