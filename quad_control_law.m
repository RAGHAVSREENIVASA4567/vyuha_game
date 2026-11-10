function [T, Ux, Uy, Uz] = quad_control_law(state, ref)
%QUAD_CONTROL_LAW  Cascaded PID control law for a 4-rotor (X-config) quadrotor.
%
% This is the "Control Law" block from the handwritten diagram:
%   [IMU, FP] --> [Control Law] --> [T, Ux, Uy, Uz] --> [M^-1] --> PWM
%
% INPUTS
%   state = [z, zdot, phi, phidot, theta, thetadot, psi, psidot]
%           measured / estimated states from IMU + state estimator ("FP")
%   ref   = [zd, zd_dot, phid, phid_dot, thetad, thetad_dot, psid, psid_dot]
%           desired trajectory from the Flight Planner ("FP")
%
% Using vectors (rather than 16 separate scalar ports) keeps the Simulink
% wiring to just two signal lines in, matching the single "[IMU,FP]" arrow
% feeding the Control Law block in the notes.
%
% OUTPUTS
%   T   - total collective thrust command
%   Ux  - pitch-axis torque command   (image: Ux = -kp(theta-thetad)-kd(thetadot-thetadot_d)-ki*int(theta-thetad))
%   Uy  - roll-axis torque command    (image: Uy = -kp(phi-phid)  -kd(phidot-phid_dot)  -ki*int(phi-phid))
%   Uz  - yaw-axis torque command     (not shown explicitly in the notes; standard PD added so the
%                                       mixing matrix in Image 2, which needs Uz, has a driving signal)
%
% NOTE ON NAMING: the handwritten notes label the pitch-torque command "Ux" and the roll-torque
% command "Uy". That convention is kept here so this file lines up 1:1 with the photographed
% derivation. If your airframe convention is the opposite, just swap Ux/Uy at the call site.
%
% This function is written to be dropped straight into a Simulink "MATLAB Function" block.
% Persistent variables hold the integral terms so it behaves correctly as a discrete-time block
% (set the block's sample time to match your control-loop rate, e.g. dt = 0.004 as in the
% reference geometric-control script).

    %% ---- Tunable gains (edit these, or promote to block mask parameters) ----
    % Altitude loop
    persistent Kp_z Kd_z Ki_z
    if isempty(Kp_z)
        Kp_z = 6.0;   Kd_z = 4.0;   Ki_z = 0.5;
    end

    % Attitude loop (roll & pitch share gains; yaw usually softer)
    persistent Kp_att Kd_att Ki_att Kp_yaw Kd_yaw
    if isempty(Kp_att)
        Kp_att = 6.0;  Kd_att = 1.2;  Ki_att = 0.05;
        Kp_yaw = 2.0;  Kd_yaw = 0.6;
    end

    % Physical / saturation parameters
    persistent m g T_max U_max dt
    if isempty(m)
        m     = 0.5;    % kg, matches mass used in the reference script
        g     = 9.8;    % m/s^2
        T_max = 20;     % N, collective thrust saturation (tune to airframe)
        U_max = 5;      % N*m, torque saturation (tune to airframe)
        dt    = 0.004;  % s, must match this block's Simulink sample time
    end

    %% ---- Persistent integrator states (anti-windup via clamping) ----
    persistent int_z int_phi int_theta
    if isempty(int_z)
        int_z = 0; int_phi = 0; int_theta = 0;
    end

    %% ---- Unpack vector inputs ----
    z     = state(1); zdot     = state(2);
    phi   = state(3); phidot   = state(4);
    theta = state(5); thetadot = state(6);
    psi   = state(7); psidot   = state(8);

    zd     = ref(1); zd_dot     = ref(2);
    phid   = ref(3); phid_dot   = ref(4);
    thetad = ref(5); thetad_dot = ref(6);
    psid   = ref(7); psid_dot   = ref(8);

    %% ---- Altitude (thrust) loop ----
    e_z    = z    - zd;
    e_zdot = zdot - zd_dot;

    int_z = int_z + e_z*dt;
    int_z = max(min(int_z, 5), -5);   % anti-windup clamp

    % Image 1: T = (m*g - Kp*(z-zd) - Kd*(zdot-zd_dot)) / cos(phi)
    % (extended with the integral term and cos(theta) since thrust projects
    %  onto the world-vertical through BOTH roll and pitch tilt)
    cos_tilt = cos(phi)*cos(theta);
    cos_tilt = sign(cos_tilt)*max(abs(cos_tilt), 0.2);  % guard against near-90 deg tilt singularity

    T = (m*g - Kp_z*e_z - Kd_z*e_zdot - Ki_z*int_z) / cos_tilt;
    T = max(min(T, T_max), 0);   % thrust can't be negative

    %% ---- Pitch loop -> Ux (per image labeling) ----
    e_theta    = theta    - thetad;
    e_thetadot = thetadot - thetad_dot;
    int_theta  = int_theta + e_theta*dt;
    int_theta  = max(min(int_theta, 2), -2);

    Ux = -Kp_att*e_theta - Kd_att*e_thetadot - Ki_att*int_theta;
    Ux = max(min(Ux, U_max), -U_max);

    %% ---- Roll loop -> Uy (per image labeling) ----
    e_phi    = phi    - phid;
    e_phidot = phidot - phid_dot;
    int_phi  = int_phi + e_phi*dt;
    int_phi  = max(min(int_phi, 2), -2);

    Uy = -Kp_att*e_phi - Kd_att*e_phidot - Ki_att*int_phi;
    Uy = max(min(Uy, U_max), -U_max);

    %% ---- Yaw loop -> Uz (PD, not explicit in the notes but required by the mixing matrix) ----
    e_psi    = psi    - psid;
    e_psidot = psidot - psid_dot;

    Uz = -Kp_yaw*e_psi - Kd_yaw*e_psidot;
    Uz = max(min(Uz, U_max), -U_max);

end
