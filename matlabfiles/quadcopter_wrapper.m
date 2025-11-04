function dxdt = quadcopter_wrapper(x, u)
% Wrapper to call the protected quadcopter model
dxdt = quadcopter_model(x, u);  % quadcopter_model is the .p file
end