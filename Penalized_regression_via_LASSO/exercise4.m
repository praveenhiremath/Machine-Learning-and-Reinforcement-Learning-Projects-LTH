%% Exercise 4: Cyclic coordinate descent solver application
clear;
close all; 
load('A1_data.mat')

lambda_vals = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
%clrs = ["#FF0000", "#A2142F", "#D95319", "#FF00FF", "#0000FF", "#0072BD", "#4DBEEE", "#00FFFF", "#00FF00", "#77AC30", "#7E2F8E"];

non_zero_w = [];
for i = 1:length(lambda_vals)
    what1 = lasso_ccd(t, X, lambda_vals(i));
    non_zero_w(i) = sum(what1 ~= 0.0);
    yinterp = Xinterp*what1;
    yestim = X*what1;


    figure 
    hold on
    plot(n, t, 's', 'MarkerSize', 10,'color', 'r')
    plot(n, yestim, '*', 'color', 'b')
    plot(ninterp, yinterp, 'color', 'k')
    set(gca, 'FontSize', 16)
    xlabel('Time')
    legend('Given data points', 'Data points reconstructed from new W', 'Interpolation')
    title(['\fontsize{24} \lambda = ', num2str(lambda_vals(i))])

    what1=[];    
    
end

non_zero_w
