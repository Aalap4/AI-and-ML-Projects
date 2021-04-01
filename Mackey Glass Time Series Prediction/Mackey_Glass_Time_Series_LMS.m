%% Mackey Glass Time Series Prediction Using Least Mean Square (LMS)
clc
clear all
close all

%% Loading Time series data
% Mackey glass series equation with the following configurations:
% b = 0.1, a = 0.2, Tau = 20, and the initial conditions x(t - Tau) = 1.2.
load X_data.mat % Data generated using mackeyglass.m
time_steps=1;
teacher_forcing=1; % recurrent ARMA modelling with forced desired input after defined time steps 
%% Training and Testing datasets
% For training
Tr=X(1:4000,1);    % Selecting a Interval of series data t = 100~2500
Xr(Tr)=X(Tr,2);      % Selecting a chuck of series data x(t)
% For testing
Ts=X(4000:5000,1);   % Selecting a Interval of series data t = 2500~3000
Xs(Ts)=X(Ts,2);      % Selecting a chuck of series data x(t)

%% LMS Parameters

eta=5e-3;       % Learning rate
M=10;            % Order of LMS filter

U=zeros(1,M+1);% Initial values of taps

W=zeros(M+1,1); % Initial weight of LMS

MSE=[];         % Initial mean squared error (MSE)

%% Learning weights of LMS (Training)
tic % start
for i=Tr(1):Tr(end)-time_steps
    U(1:end-1)=U(2:end);    % Shifting of tap window
    
    if (teacher_forcing==1)
        if rem(i,time_steps)==0 || (i==Tr(1))
            U(end)=Xr(i);           % Input (past/current samples)
        else
            U(end)=Y(i-1);          % Input (past/current samples)          
        end
    else
        U(end)=Xr(i);           % Input (past/current samples)
    end
 
    Y(i)=W'*U';             % Predicted output
    e(i)=Xr(i+time_steps)-Y(i);        % Error in predicted output

    W=W+eta*e(i)*U';     % Weight update rule of LMS
    
    %E(i)=mean(e(Tr(1):i).^2);   % Current mean squared error (MSE)
    E(i)=e(i).^2;   % Current mean squared error (MSE)
end
training_time=toc; % total time including training and calculation of MSE

%% Prediction of a next outcome of series using previous samples (Testing)
tic % start
%U=U*0;  % Reinitialization of taps (optional)
for i=Ts(1):Ts(end)-time_steps+1
    U(1:end-1)=U(2:end);    % Shifting of tap window

    if (teacher_forcing==1)
        if rem(i,time_steps)==0 || (i==Ts(1))
            U(end)=Xs(i);           % Input (past/current samples)
        else
            U(end)=Y(i-1);          % Input (past/current samples)          
        end
    else
        U(end)=Xs(i);           % Input (past/current samples)
    end
    
    
    Y(i)=W'*U';             % Calculating output (future value)
    e(i)=Xs(i+time_steps-1)-Y(i);        % Error in predicted output

    %E(i)=mean(e(Ts(1):i).^2);   % Current mean squared error (MSE)
    E(i)=e(i).^2;   % Current mean squared error (MSE)
end
testing_time=toc; % total time including testing and calculation of MSE

%% Results

plot(Tr,10*log10(E(Tr)));   % MSE curve
hold on
plot(Ts(1:end-time_steps+1),10*log10(E(Ts(1:end-time_steps+1))),'r');   % MSE curve
grid minor
% legend('Training','Testing');
title('Cost Function');
xlabel('Iterations (samples)');
ylabel('Mean Squared Error (MSE)');
legend('Training Phase','Test Phase');
figure
plot(Tr(2*M:end),Xr(Tr(2*M:end)));      % Actual values of mackey glass series
hold on
plot(Tr(2*M:end),Y(Tr(2*M:end))','r')   % Predicted values during training

plot(Ts,Xs(Ts),'--b');        % Actual unseen data
hold on
plot(Ts(1:end-time_steps+1),Y(Ts(1:end-time_steps+1))','--r');  % Predicted values of mackey glass series (testing)
xlabel('Time: t');
ylabel('Output: Y(t)');
% legend('Actual (Training)','Predicted (Training)','Actual (Testing)','Predicted (Testing)');
title('Mackey Glass Time Series Prediction Using Least Mean Square (LMS)')
ylim([min(Xs)-0.5, max(Xs)+0.5])

legend('Training Phase (desired)','Training Phase (predicted)','Training Phase (desired)','Test Phase (predicted)');

mitr=10*log10(mean(E(Tr)));  % Minimum MSE of training
mits=10*log10(mean(E(Ts(1:end-time_steps+1))));  % Minimum MSE of testing

display(sprintf('Total training time is %.5f, \nTotal testing time is %.5f \nMSE value during training %.3f (dB),\nMSE value during testing %.3f (dB)', ...
training_time,testing_time,mitr,mits));

