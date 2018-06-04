function [s, p1] = generate_bernoulli_sequence(L, pJump)
% [s, gen_prob] = generate_sequence(L, pJump)
% 
% This function generates a random binary sequence (s), based on a "jumping" 
% bernoulli probability. 
% The sequence length is L, and the probability of a jump in the bernoulli
% parameter at any trial is pJump (if not specified, this is set to 
% 1/75 by default).
% At any trial, the identity of the item ("1" or "2") is selected randomly 
% based on predefined bernoulli parameter: p(1). This parameter and the 
% moment it jumps, is itself sampled randomly by this function. 
% The output gen_prob returns the values of the bernoulli parameter.
%
% Florent Meyniel, Jun. 2018

% PARAMETERS
% ==========
LJumpMax     = 250;      % Maximum length for a stable period
MinOddChange = 4;        % fold change (from pre to post jump) of odd for at least 1 transition
pMin         = 0.1;      % minimum value for transition probabilities
pMax         = 0.9;      % maximum value for transition probabilities
if ~exist('pJump', 'var')
    pJump    = 1/75;
end

% INITIALIZATION
% ==============

% Initialize random generators
try
    s = RandStream('mt19937ar','Seed','shuffle');
    RandStream.setGlobalStream(s);
catch
    rand('twister',sum(100*clock))
end

% % check whether heaviside is known
% try
%     heaviside(1);
% catch
%     heaviside = @(x) double(x > 0);
% end

% check whether heaviside is known (Sebastien Demortain's modif to make it
% work)
is_heaviside_defined = exist('heaviside');
if (is_heaviside_defined~=0)
    heaviside = @(x) double(x > 0);
end

% PREPARE RANDOM SEQUENCE
% =======================

% DEFINE JUMP OCCURENCE
% Define jumps occurence with a geometric law of parameter pJump
% CDF = 1-(1-pJump)^k => k=log(1-CDF)/log(1-pJump)
% the geometric distribution can be sampled by sampling uniformly from CDF.
SubL = [];
while sum(SubL) < L
    RandGeom = Inf;
    while RandGeom >= LJumpMax
        RandGeom = round(log(1-rand) / log(1-pJump));
    end
    SubL = [SubL, RandGeom];
end

% DEFINE TRANSITION PROBABILITIES
tmp_p1 = zeros(1, length(SubL));
for kJump = 1:length(SubL)
    
    % Initialize at random values (with restrictions)
    if kJump == 1
        tmp_p1(kJump) = rand*(pMax-pMin) + pMin;
    else
        while true
            tmp_p1(kJump) = rand*(pMax-pMin) + pMin;
            
            % compute transition odd change from pre to post jump
            oddChange = (tmp_p1(kJump-1)/(1-tmp_p1(kJump-1))) ...
                / (tmp_p1(kJump)/(1-tmp_p1(kJump)));
            
            % take this value if the change is sufficiently large
            if abs(log(oddChange)) > log(MinOddChange) || ...
                    abs(log(oddChange)) > log(MinOddChange)
                break
            end
        end
    end
end

% assign transition probs value to each trial
p1 = zeros(1, L);
p1(1:SubL(1)) = tmp_p1(1);
for kJump = 2:(length(SubL)-1)
    p1( sum(SubL(1:(kJump-1)))+1 : sum(SubL(1:kJump)) ) = tmp_p1(kJump);
end
p1( sum(SubL(1:end-1))+1 : end) = tmp_p1(end);

% GENERATE SEQUENCE ACCORDING TO THESE TRANSITION PROBS
s = heaviside(rand(1, L) - p1) + 1;

