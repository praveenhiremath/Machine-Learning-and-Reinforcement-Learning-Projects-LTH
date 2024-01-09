%% Exercise 6 : Compute optimal lambda-value for all frames of multiframe audio excerpt "Ttrain", 
% using K-fold validation scheme implemented in the file "multiframe_lasso_cv.m"
clear;
close all; 
load('A1_data.mat')

max_lambda = 10 %max(abs(Xaudio'*Ttrain));    
min_lambda = 0.0001;
tot_lambdas = 100;
Folds = 10;
lambdas = exp(linspace(log(max_lambda),log(min_lambda),tot_lambdas));


[opt_w, lambdaopt, RMSEval, RMSEest] = multiframe_lasso_cv(Ttrain, Xaudio, lambdas, Folds);
save('model_from_exercise_6.mat', 'opt_w', 'lambdaopt', 'RMSEval', 'RMSEest')

figure(1)
hold on
%Plotting RMSE on validation dataset vs log(lambda)-values
plot(log(lambdas), RMSEval, '-x','MarkerSize', 10, 'color','m')
%Plotting RMSE on estimation dataset vs log(lambda)-values
plot(log(lambdas), RMSEest, '-o', 'MarkerSize', 5, 'color','b')

xline(log(lambdaopt), '-', {});
text(-5.0,0.01,{'\lambda = ',num2str(lambdaopt)},'FontSize',20);
set(gca, 'FontSize', 24)
legend('RMSE on validation', 'RMSE on estimation', 'Optimal \lambda-value (all frames)')
xlabel('log(\lambda)')
title(['\fontsize{24} Exercise 6: RMSE versus log(\lambda)'])
set(gcf,'Position',[10 1000 1500 1000])
saveas(gcf,'task6_errors_vs_log_lambda.png')


