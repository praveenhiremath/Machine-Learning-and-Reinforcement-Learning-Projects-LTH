function [values, policy, nbr_pol_iter, nbr_pol_eval] = policy_iteration(pol_eval_tol, next_state_idxs, rewards, gamm)
%
% Code may be changed in this function, but only where it states that it is 
% allowed to do so. You need to implement policy iteration in this
% function.
%
% Function to run policy iteration to learn an optimal policy. Note that
% this implementation assumes eating apple is also a terminal state (and 
% not only hitting a wall / the body of the snake). Think about why this 
% is OK, i.e., why it will give rise to an optimal policy also for the 
% Snake game in which eating an apple is non-terminal (see also exercise 4).
%
% Input:
%
% pol_eval_tol    - Policy evaluation stopping tolerance.
% next_state_idxs - nbr_states-by-nbr_actions matrix; each entry of this 
%                   matrix is an integer in {-1, 0, 1, 2, ..., nbr_states}.
%                   In particular, the ith row of next_state_idxs gives
%                   the state indexes of taking the left, forward and right 
%                   actions. The only exceptions to this is if an action leads 
%                   to any terminal state; if an action leads to death, then
%                   the corresponding entry in next_state_idxs is 0; if an
%                   action leads to eating an apple, then the corresponding
%                   entry in next_state_idxs is -1.
% rewards         - Struct of the form struct('default', x, 'apple', y, 'death', z)
%                   Here x refers to the default reward, which is received
%                   when the snake only moves without dying and without
%                   eating an apple; y refers to the reward obtained when
%                   eating an apple; and z refers to the reward obtained when
%                   dying.
% gamm            - Discount factor in [0,1].
%
% Output:
%
% values       - 1-by-nbr_states vector; will after successful policy
%                iteration contain optimal values for all the non-terminal 
%                states.
% policy       - 1-by-nbr_states vector; will after successful policy
%                iteration contain optimal actions to take for all the 
%                non-terminal states.
% nbr_pol_iter - The number of iterations that the policy iteration runs
%                for; may be used e.g. for diagnostic purposes.
%                   
% Bugs, ideas etcetera: send them to the course email.

% Get number of non-terminal states and actions.
[nbr_states, nbr_actions] = size(next_state_idxs);

% Arbitrary initialization of values and policy.
values = randn(1, nbr_states);  
policy = randi(3, 1, nbr_states); % policy is size 1-by-nbr_states
                                  % the entries of policy are 1, 2 or 3
                                  % selected uniformly at random

% Counters over number of policy iterations and policy evaluations,
% for possible diagnostic purposes.
nbr_pol_iter = 0;
nbr_pol_eval = 0;

% This while-loop runs the policy iteration.
while 1
    
    % Policy evaluation.
    while 1
        
        Delta = 0;
        for state_idx = 1 : nbr_states
            % FILL IN POLICY EVALUATION WITHIN THIS LOOP.
            val = values(state_idx);
            act = policy(state_idx);
            next_s = next_state_idxs(state_idx, act);

            if and((next_s ~= -1), (next_s ~= 0))  % Snake moves without getting apple or dying
                values(state_idx) = rewards.default + values(next_s)*gamm;
            elseif next_s == 0  % Snake dies
                values(state_idx) = rewards.death;
            else   % Snake eates an apple
                values(state_idx) = rewards.apple;
            end
            Delta = max(abs(val-values(state_idx)), Delta);
        end
        
        % Increase nbr_pol_eval counter.
        nbr_pol_eval = nbr_pol_eval + 1;
        
        % Check for policy evaluation termination.
        if Delta < pol_eval_tol
            break;
        else
            disp(['Delta: ', num2str(Delta)])
        end
    end
    
    % Policy improvement.
    policy_stable = true; 
    for state_idx = 1 : nbr_states
        % FILL IN POLICY IMPROVEMENT WITHIN THIS LOOP.
        %next_s_id = next_state_idxs(state_idx);
        prev_act = policy(state_idx);
        vals = zeros(length(next_state_idxs(state_idx,:)),1);
        for i = 1:length(vals)
            if and((next_state_idxs(state_idx,i) ~= -1),(next_state_idxs(state_idx,i) ~= 0))
                vals(i) = rewards.default + values(next_state_idxs(state_idx,i))*gamm;
            elseif next_state_idxs(state_idx,i) == 0
                vals(i) = rewards.death;
            else
                vals(i) = rewards.apple;
            end
        end
        [num, ind] = max(vals);
        policy(state_idx) = ind;
        if (prev_act ~= policy(state_idx))
            policy_stable = false;
        end

    end
    
    % Increase the number of policy iterations .
    nbr_pol_iter = nbr_pol_iter + 1;
    
    % Check for policy iteration termination (terminate if and only if the
    % policy is no longer changing, i.e. if and only if the policy is
    % stable).
    if policy_stable
        break;
    end
end
end
