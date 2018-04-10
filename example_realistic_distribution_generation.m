clear all; close all; clc;

% Ideal observer for specific sequences
p = genpath('C:\Users\Sébastien\Documents\DTU\Master thesis\code');
addpath(p);
load('sequence-genprob_4.mat')

%% EXAMPLE 1: COMPUTE THE IDEAL OBSERVER WITH JUMPS (HMM)
%  ======================================================

pJump = 1/75; % Value by default

% Set parameters
in.s            = s;                % sequence
in.learned      = 'transition';     % estimate transition
in.jump         = 1;                % estimate with jumps
in.mode         = 'HMM';            % use the HMM (not sampling) algorithm
in.opt.pJ       = pJump;            % a priori probability that a jump occur at each outcome
n               = 50;               % resolution of the univariate probability grid
in.opt.pgrid    = linspace(0,1,n);  % estimation probability grid
in.opt.Alpha0   = ones(n)/(n^2);    % uniform prior on transition probabilities
in.verbose      = 1;                % to check that no default values are used.

% Compute the observer
out.HMM = IdealObserver(in);
