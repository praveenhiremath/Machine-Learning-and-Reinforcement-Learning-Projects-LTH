function [dldX, dldW, dldb] = fully_connected_backward(X, dldY, W, b)
    % Inputs
    %   X - The input variable. The size might vary, but the last dimension
    %       tells you which element in the batch it is.
    %   dldY - The partial derivatives of the loss with respect to the
    %       output variable Y. The size of dldY is the same as for Y as
    %       computed in the forward pass.
    %   W  - The weight matrix
    %   b  - The bias vector
    %
    % Outputs
    %    dldX - Gradient backpropagated to X
    %    dldW - Gradient backpropagated to W
    %    dldb - Gradient backpropagated to b
    %
    % All gradients should have the same size as the variable. That is,
    % dldX and X should have the same size, dldW and W the same size and dldb
    % and b the same size.
    sz = size(X);
    batch = sz(end);
    features = prod(sz(1:end-1));

    % We reshape the input vector so that all features for a single batch
    % element are in the columns. X is now as defined in the assignment.
    X = reshape(X, [features, batch]);
    
    assert(size(W, 2) == features, ...
        sprintf('Expected %d columns in the weights matrix, got %d', features, size(W,2)));
    assert(size(W, 1) == numel(b), 'Expected as many rows in W as elements in b');
    
    % Implement it here.
    % note that dldX should have the same size as X, so use reshape
    % as suggested.
    
    % For Backward propagation implementation
    dldb = sum(dldY, 2);
    
    dldX = W'*dldY;
    dldX = reshape(dldX, sz);
    dldW = dldY*X';
end
