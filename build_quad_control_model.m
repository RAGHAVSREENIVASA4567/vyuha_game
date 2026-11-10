%BUILD_QUAD_CONTROL_MODEL
% Programmatically builds quad_control_model.slx, reproducing the block
% diagram in the handwritten notes:
%
%   [IMU]  --\
%             >-- [Control Law] --[T,Ux,Uy,Uz]--> [M^-1 / PWM] --> PWM_Out
%   [FP]   --/                         |
%                                      +--> Scope (T, Ux, Uy, Uz)
%
% REQUIREMENTS: MATLAB with Simulink AND Stateflow (Stateflow ships with
% the "MATLAB Function" block). Run this script from a folder that also
% contains quad_control_law.m and quad_mixing_pwm.m.
%
% USAGE:
%   >> build_quad_control_model
% This creates and opens quad_control_model.slx in the current folder.

modelName = 'quad_control_model';

% ---- Fresh model ----
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
if exist([modelName '.slx'], 'file')
    delete([modelName '.slx']);
end
new_system(modelName);
open_system(modelName);

%% ---- Inports: State (from IMU) and Reference (from Flight Planner) ----
add_block('simulink/Sources/In1', [modelName '/State_IMU']);
set_param([modelName '/State_IMU'], 'Position', [40 60 90 90], ...
    'PortDimensions', '8', 'OutDataTypeStr', 'double');

add_block('simulink/Sources/In1', [modelName '/Reference_FP']);
set_param([modelName '/Reference_FP'], 'Position', [40 160 90 190], ...
    'PortDimensions', '8', 'OutDataTypeStr', 'double');

%% ---- MATLAB Function block: Control Law ----
add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [modelName '/Control_Law']);
set_param([modelName '/Control_Law'], 'Position', [180 90 320 220]);

% Inject the control-law source into the MATLAB Function block's script.
rt = sfroot;
ctrlChart = rt.find('-isa', 'Stateflow.EMChart', ...
    'Path', [modelName '/Control_Law']);
ctrlChart.Script = fileread('quad_control_law.m');

%% ---- Mux: combine [T,Ux,Uy,Uz] scalar outputs into one vector ----
add_block('simulink/Signal Routing/Mux', [modelName '/Mux_TUxUyUz']);
set_param([modelName '/Mux_TUxUyUz'], 'Position', [380 100 400 200], ...
    'Inputs', '4');

%% ---- MATLAB Function block: Mixing matrix + rotor speed + PWM ----
add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [modelName '/Mixing_PWM']);
set_param([modelName '/Mixing_PWM'], 'Position', [460 110 620 190]);

mixChart = rt.find('-isa', 'Stateflow.EMChart', ...
    'Path', [modelName '/Mixing_PWM']);
mixChart.Script = fileread('quad_mixing_pwm.m');

%% ---- Outport: PWM commands to the 4 ESCs/motors ----
add_block('simulink/Sinks/Out1', [modelName '/PWM_Out']);
set_param([modelName '/PWM_Out'], 'Position', [680 130 730 160]);

%% ---- Scopes for T/Ux/Uy/Uz and PWM (useful during tuning) ----
add_block('simulink/Sinks/Scope', [modelName '/Scope_TUxUyUz']);
set_param([modelName '/Scope_TUxUyUz'], 'Position', [400 240 440 280], ...
    'NumInputPorts', '1');

add_block('simulink/Sinks/Scope', [modelName '/Scope_PWM']);
set_param([modelName '/Scope_PWM'], 'Position', [680 220 720 260], ...
    'NumInputPorts', '1');

%% ---- Wiring ----
add_line(modelName, 'State_IMU/1',     'Control_Law/1');
add_line(modelName, 'Reference_FP/1',  'Control_Law/2');

% Control_Law has 4 scalar outputs [T, Ux, Uy, Uz] -> Mux inputs 1..4
add_line(modelName, 'Control_Law/1', 'Mux_TUxUyUz/1');
add_line(modelName, 'Control_Law/2', 'Mux_TUxUyUz/2');
add_line(modelName, 'Control_Law/3', 'Mux_TUxUyUz/3');
add_line(modelName, 'Control_Law/4', 'Mux_TUxUyUz/4');

add_line(modelName, 'Mux_TUxUyUz/1',  'Mixing_PWM/1');
add_line(modelName, 'Mux_TUxUyUz/1',  'Scope_TUxUyUz/1');

add_line(modelName, 'Mixing_PWM/1',   'PWM_Out/1');
add_line(modelName, 'Mixing_PWM/1',   'Scope_PWM/1');

%% ---- Model-wide sample time (must match dt used inside quad_control_law.m) ----
set_param(modelName, 'Solver', 'FixedStepDiscrete', 'FixedStep', '0.004');

%% ---- Save ----
save_system(modelName);
fprintf('Built and saved %s.slx\n', modelName);
