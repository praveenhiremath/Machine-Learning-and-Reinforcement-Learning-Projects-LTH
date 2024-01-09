function [state_action_feats, prev_grid, prev_head_loc] = extract_state_action_features(prev_grid, grid, prev_head_loc, nbr_feats)
%
% Code may be changed in this function, but only where it states that it is 
% allowed to do so.
%
% Function to extract state-action features, based on current and previous
% grids (game screens).
%
% Input:
%
% prev_grid     - Previous grid (game screen), N-by-N matrix. If initial
%                 time-step: prev_grid = grid, else: prev_grid != grid.
% grid          - Current grid (game screen), N-by-N matrix. If initial
%                 time-step: prev_grid = grid, else: prev_grid != grid.
% prev_head_loc - The previous location of the head of the snake (from the 
%                 previous time-step). If initial time-step: Assumed known,
%                 else: inferred in function "update_snake_grid.m" (so in
%                 practice it will always be known in this function).
% nbr_feats     - Number of state-action features per action. Set this 
%                 value appropriately in the calling script "snake.m", to
%                 match the number of state-action features per action you
%                 end up using.
%
% Output:
%
% state_action_feats - nbr_feats-by-|A| matrix, where |A| = number of
%                      possible actions (|A| = 3 in Snake), and nbr_feats
%                      is described under "Input" above. This matrix
%                      represents the state-action features extracted given
%                      the current and previous grids (game screens).
% prev_grid          - The previous grid as seen from one step in the
%                      future, i.e., prev_grid is set to the input grid.
% prev_head_loc      - The previous head location as seen from one step
%                      in the future, i.e., prev_head_loc is set to the
%                      current head location (the current head location is
%                      inferred in the code below).
%
% Bugs, ideas etcetera: send them to the course email.

% --------- DO NOT CHANGE ANYTHING BELOW UNLESS OTHERWISE NOTIFIED! -------

% Extract grid size.
N = size(grid, 1);

% Initialize state_action_feats to nbr_feats-by-3 matrix.
state_action_feats = nan(nbr_feats, 3);

% Based on how grid looks now and at previous time step, infer head
% location.
change_grid = grid - prev_grid;
prev_grid   = grid; % Used in later calls to "extract_state_action_features.m"

% Find head location (initially known that it is in center of grid).
if nnz(change_grid) > 0 % True, except in initial time-step
    [head_loc_m, head_loc_n] = find(change_grid > 0);
else % True only in initial time-step
    head_loc_m = round(N / 2);
    head_loc_n = round(N / 2);
end
head_loc = [head_loc_m, head_loc_n];

% Previous head location.
prev_head_loc_m = prev_head_loc(1);
prev_head_loc_n = prev_head_loc(2);

% Infer current movement directory (N/E/S/W) by looking at how current and previous
% head locations are related
if prev_head_loc_m == head_loc_m + 1 && prev_head_loc_n == head_loc_n     % NORTH
    movement_dir = 1;
elseif prev_head_loc_m == head_loc_m && prev_head_loc_n == head_loc_n - 1 % EAST
    movement_dir = 2;
elseif prev_head_loc_m == head_loc_m - 1 && prev_head_loc_n == head_loc_n % SOUTH
    movement_dir = 3;
else                                                                      % WEST
    movement_dir = 4;
end

% The current head_loc will at the next time-step be prev_head_loc.
prev_head_loc = head_loc;

% ------------- YOU MAY CHANGE SETTINGS BELOW! --------------------------

% HERE BEGINS YOUR STATE-ACTION FEATURE ENGINEERING. THE CODE BELOW IS 
% ALLOWED TO BE CHANGED IN ACCORDANCE WITH YOUR CHOSEN FEATURES. 
% Some skeleton code is provided to help you get started. Also, have a 
% look at the function "get_next_info" (see bottom of this function).
% You may find it useful.

for action = 1 : 3 % Evaluate all the different actions (left, forward, right).
    
    % Feel free to uncomment below line of code if you find it useful.
    %[next_head_loc, next_move_dir] = get_next_info(action, movement_dir, head_loc);
    
    % Replace this to fit the number of state-action features per features
    % you choose (3 are used below), and of course replace the randn() 
    % by something more sensible.


    % Feature 1: Distance to apple (pixel '-1'). Weight is '+1'
    max_pos_l1_dis = norm(size(grid),1);  % Tried L1 and L2 norms: L1 gives better score.
    [next_s_loc, next_s_dir] = get_next_info(action, movement_dir, head_loc);  % Getting head position and direction in next state 

    [minusones_rows, minusones_cols] = find(grid == -1);  % Location of apple
    minus_ones_pix = [minusones_rows, minusones_cols];
    dist_minusones = norm(next_s_loc - minus_ones_pix,1);   % Tried L1 and L2 norms: L1 gives better score
    state_action_feats(1, action) = (dist_minusones/max_pos_l1_dis); % Normalizing

%   Feature 2: Next move landing in pixel of '0' or '-1'? Weight is '-1'
    if (grid(next_s_loc(1),next_s_loc(2)) ~= 1)  
        state_action_feats(2, action) = 1;   
    else
        state_action_feats(2, action) = -1;
    end

%{
% Feature 3: Distance of snake head from pixels of 1(snake/wall). weight is '-1'
% Feature promoting maximum distance between head and pixels of '1'. 
    [ones_rows, ones_cols] = find(grid == 1); %snake and wall  
    dist_ind_ones = zeros(length(ones_rows));
    for i = 1:length(ones_rows)
        dist_ind_ones(i) = norm(next_s_loc-[ones_rows(i), ones_cols(i)],1); % Tried L1 and L2 norms
    end

    [max_dist_ones,index] = max(dist_ind_ones,[],'all');
    dist_ones = max_dist_ones;
    state_action_feats(3,action) = (dist_ones/max_pos_l1_dis); %Normalizing



% Feature 4: Looking a bit more ahead to avoid pixels of '1' (snake/wall). weight is '-1'.
    desired_pix = [0 -1];
%   Testing head running into pixels of '1'.
    if ((next_s_loc(2)+1<(N-1)) & (next_s_loc(1)+1<(N-1))) & ((next_s_loc(2)-1>0) & (next_s_loc(1)-1>0))
        if ((ismember(grid(next_s_loc(1), next_s_loc(2)+1),desired_pix)) | (ismember(grid(next_s_loc(1)+1, next_s_loc(2)+1),desired_pix)) | (ismember(grid(next_s_loc(1)-1, next_s_loc(2)+1),desired_pix)) | (ismember(grid(next_s_loc(1)+1, next_s_loc(2)),desired_pix)))
            state_action_feats(4, action) = 1;
        elseif ((ismember(grid(next_s_loc(1), next_s_loc(2)-1),desired_pix)) | (ismember(grid(next_s_loc(1)+1, next_s_loc(2)-1),desired_pix)) | (ismember(grid(next_s_loc(1)-1, next_s_loc(2)-1),desired_pix))) 
            state_action_feats(4, action) = 1;
        else
            state_action_feats(4, action) = -1;
        end
    else
        state_action_feats(4, action) = -1;
    end



% Feature 5: to avoid very narrow snake configuration.
% Copying grid info
    inner_grid = grid;
    size(inner_grid);
    inner_grid(:,1) = [];
    inner_grid(:,end) = [];
    inner_grid(1,:) = [];
    inner_grid(end,:) = [];
    size(inner_grid);  % This is 28X28
    [snake_i, snake_j] = find(inner_grid == 1);
    row_dist = max(snake_i) - min(snake_i);
    col_dist = max(snake_j) - min(snake_j);
    if and((and(col_dist == 0, row_dist == 10)),(and(col_dist == 0, row_dist == 10)))
        state_action_feats(5, action) = 1;  % straight config 
    elseif (or(row_dist == 1, col_dist == 1))
        state_action_feats(5, action) = 0; % very narrow config
    else
        state_action_feats(5, action) = 1; % Not too narrow 
    end
%}

    %state_action_feats(1, action) = randn();
    %state_action_feats(2, action) = randn();
    %state_action_feats(3, action) = randn();
    % ... and so on ...
end
end

%
% DO NOT CHANGE ANYTHING IN THE FUNCTION get_next_info BELOW!
%
function [next_head_loc, next_move_dir] = get_next_info(action, movement_dir, head_loc)
% Function to infer next haed location and movement direction

% Extract relevant stuff
head_loc_m = head_loc(1);
head_loc_n = head_loc(2);

if movement_dir == 1 % NORTH
    if action == 1     % left
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4; 
    elseif action == 2 % forward
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    else               % right
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    end
elseif movement_dir == 2 % EAST
    if action == 1
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    elseif action == 2
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    else
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    end
elseif movement_dir == 3 % SOUTH
    if action == 1
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    elseif action == 2
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    else
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4;
    end
else % WEST
    if action == 1
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    elseif action == 2
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4;
    else
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    end
end
next_head_loc = [next_head_loc_m, next_head_loc_n];
end



