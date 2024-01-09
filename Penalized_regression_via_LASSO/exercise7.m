%% Exercise 7: Denoising the test dataset "Ttest"

clear;
close all;

load('A1_data.mat')
load('model_from_exercise_6.mat')

soundsc(Ttest,fs)
Yclean = lasso_denoise(Ttest,Xaudio,lambdaopt);
soundsc(Yclean,fs)
save('denoised_audio','Yclean','fs')

Yclean2 = lasso_denoise(Ttest,Xaudio,0.01);
soundsc(Yclean2,fs)
save('denoised_audio_0_01','Yclean2','fs')


Yclean3 = lasso_denoise(Ttest,Xaudio,0.02);
soundsc(Yclean3,fs)
save('denoised_audio_0_02','Yclean3','fs')

Yclean4 = lasso_denoise(Ttest,Xaudio,0.04);
soundsc(Yclean4,fs)
save('denoised_audio_0_04','Yclean4','fs')


Yclean5 = lasso_denoise(Ttest,Xaudio,0.1);
soundsc(Yclean5,fs)
save('denoised_audio_0_1','Yclean5','fs')