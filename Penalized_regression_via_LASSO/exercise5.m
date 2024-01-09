%% Exercise 5: K-fold cross-validation scheme application
clear;
close all; 
load('A1_data.mat')

max_lambda = max(abs(X'*t));    %100; %max(abs(X'*t));
min_lambda = 0.01;
tot_lambdas = 100;
Folds = 10;
lambdas = exp(linspace(log(max_lambda),log(min_lambda),tot_lambdas));


[opt_w, lambdaopt, RMSEval, RMSEest] = lasso_cv(t, X, lambdas, Folds);

%% 5.1: Plot of RMSE on validation and estimation dataset vs log(lambda)-values
figure(1)
hold on

%Plotting RMSE on validation datasset
plot(log(lambdas), RMSEval, '-x','color','m','MarkerSize',10)
%Plotting RMSE on estimation datasset
plot(log(lambdas), RMSEest, '-o','color','b','MarkerSize',5)
%Plotting vertical line at optimal lambda value
xline(log(lambdaopt), '-', {});
opt_lam_text = text(0.9,3.0,{'\lambda = ',num2str(lambdaopt)},'FontSize',20);

set(gca, 'FontSize', 24)
legend('RMSE on validation', 'RMSE on estimation', 'Optimal \lambda-value')
xlabel('log(\lambda)')
title(['\fontsize{24} Exercise 5.1: RMSE versus log(\lambda)'])
set(gcf,'Position',[10 1000 1500 1000])
saveas(gcf,'task5_errors_vs_log_lambda.png')

%% 5.2: Plot of reconstructed data points for the optimized value of lambda
figure(2)
hold on
plot(n, t, 's', 'MarkerSize', 10,'color', 'r')
plot(n, X*opt_w, '*', 'color', 'b')
plot(ninterp, Xinterp*opt_w, 'color', 'k')
set(gca, 'FontSize', 24)
xlabel('Time indices n')
legend('Given data points', 'Data points reconstructed from new W', 'Line fit to reconstructed data point')
title(['\fontsize{24} Exercise 5.2: Optimal \lambda = ', num2str(lambdaopt)])
set(gcf,'Position',[10 1000 1500 1000])
saveas(gcf,'task5_recon_data_with_optimal_lambda.png')
