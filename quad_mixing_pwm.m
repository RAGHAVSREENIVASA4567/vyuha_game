function PWM = quad_mixing_pwm(cmd)
%QUAD_MIXING_PWM  Inverse mixing matrix + rotor-speed/PWM conversion for a 4-rotor X quadrotor.
%
% This is the "M^-1" + "PWM" block from the handwritten diagram:
%   [T, Ux, Uy, Uz] --> [M^-1] --> [M1..M4] --> PWM
%
% INPUT
%   cmd = [T; Ux; Uy; Uz]   (the Control Law block's output vector)
% OUTPUT
%   PWM = [PWM1; PWM2; PWM3; PWM4]   one command per rotor
%
% Forward mixing matrix (as photographed in Image 2), mapping individual rotor
% thrusts M1..M4 to the vehicle's total thrust/torque [T; Ux; Uy; Uz]:
%
%   [T ]   [ 1    1    1    1 ] [M1]
%   [Ux] = [-cf   cf   cf  -cf] [M2]
%   [Uy]   [ cf   cf  -cf  -cf] [M3]
%   [Uz]   [-cd   cd  -cd   cd] [M4]
%
% where:
%   cf - effective moment arm * thrust coefficient (roll/pitch torque per unit thrust differential)
%   cd - yaw drag/torque coefficient (reaction torque per unit thrust, from rotor drag)
%
% and each rotor's thrust obeys  M_i = C_T * omega_i^2   (Image 2, top-left equation).
%
% The CONTROL LAW gives us the desired [T;Ux;Uy;Uz]. This block inverts the mixing
% matrix to solve for the 4 individual rotor thrusts, converts each to a required
% rotor speed, then maps rotor speed to a normalized PWM command.

    %% ---- Physical / mixing parameters (tune to your airframe) ----
    persistent cf cd CT PWM_min PWM_max omega_max
    if isempty(cf)
        cf        = 0.17;   % N*m per N of thrust differential (arm_length * geometry factor)
        cd        = 0.02;   % N*m per N of thrust (yaw reaction coefficient)
        CT        = 3e-6;   % thrust coefficient: M_i = CT * omega_i^2
        PWM_min   = 1000;   % typical ESC PWM range (microseconds)
        PWM_max   = 2000;
        omega_max = 900;    % rad/s, max rotor speed corresponding to PWM_max
    end

    %% ---- Inverse mixing matrix ----
    Mix = [ 1    1    1    1;
           -cf   cf   cf  -cf;
            cf   cf  -cf  -cf;
           -cd   cd  -cd   cd];

    rotor_thrust = Mix \ cmd(:);   % [M1; M2; M3; M4] = Mix^-1 * [T;Ux;Uy;Uz]

    %% ---- Thrust -> rotor speed -> PWM, with saturation ----
    rotor_thrust = max(rotor_thrust, 0);        % rotors can't produce negative thrust
    omega = sqrt(rotor_thrust ./ CT);           % M_i = CT*omega_i^2  =>  omega_i = sqrt(M_i/CT)
    omega = min(omega, omega_max);

    PWM = PWM_min + (omega ./ omega_max) .* (PWM_max - PWM_min);
    PWM = max(min(PWM, PWM_max), PWM_min);

end
