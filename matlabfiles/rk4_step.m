function [x_next, dxdt_avg] = rk4_step(x, u, dt)
%RK4_STEP Perform a single RK4 integration step using quadcopter_model.
%   x: 12x1 state vector
%   u: 4x1 control input (held constant over the step)
%   dt: time step
%   Returns:
%     x_next: 12x1 next state
%     dxdt_avg: 12x1 RK4 averaged derivative

% Ensure column vectors
x = x(:);
u = u(:);

k1 = quadcopter_model(x, u);
k2 = quadcopter_model(x + 0.5*dt*k1, u);
k3 = quadcopter_model(x + 0.5*dt*k2, u);
k4 = quadcopter_model(x + dt*k3, u);

dxdt_avg = (k1 + 2*k2 + 2*k3 + k4) / 6.0;
x_next = x + dt * dxdt_avg;

end
