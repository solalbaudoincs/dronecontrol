function [X, DX] = simulate_rk4(x0, U, dt, tau)
%SIMULATE_RK4 Run N-step simulation with input LPF and RK4 inside MATLAB.
%   x0: 12x1 initial state
%   U:  Tx4 control inputs (unfiltered); if Tx1, it's broadcast to 4 motors
%   dt: time step (scalar)
%   tau: input LPF time constant (seconds)
% Returns:
%   X:  (T+1)x12 state trace (rows are timesteps)
%   DX: T x12 RK4-averaged derivatives per step

% Normalize shapes and types
x = x0(:);
T = size(U, 1);
if size(U,2) == 1
    U = repmat(U, 1, 4);
end

% Preallocate using like-types
X = zeros(T+1, numel(x), 'like', x);
DX = zeros(T,   numel(x), 'like', x);
X(1,:) = x(:).';

% LPF state and alpha
% Allocate filter state as numeric like x to avoid class issues
filt = zeros(1, size(U,2), 'like', x);
if tau <= 0
    alpha = 1.0;
else
    alpha = dt / (tau + dt);
end

for t = 1:T
    % Input filtering and saturation
    filt = filt + alpha * (U(t,:) - filt);
    voltage = 10 * tanh(filt);
    ucol = voltage(:);

    % RK4 with constant input over the step
    k1 = quadcopter_model(x, ucol);
    k2 = quadcopter_model(x + 0.5*dt*k1, ucol);
    k3 = quadcopter_model(x + 0.5*dt*k2, ucol);
    k4 = quadcopter_model(x + dt*k3, ucol);

    dxdt_avg = (k1 + 2*k2 + 2*k3 + k4) / 6.0;
    x = x + dt * dxdt_avg;

    X(t+1,:) = x(:).';
    DX(t,:) = dxdt_avg(:).';
end

end
