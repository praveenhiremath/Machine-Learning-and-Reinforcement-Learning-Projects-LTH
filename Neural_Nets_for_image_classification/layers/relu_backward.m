function dldX = relu_backward(X, dldY)
    dldX = dldY.*heaviside(X);
end
