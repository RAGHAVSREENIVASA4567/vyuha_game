# Quadrotor Control Law → Simulink Model

Three files, built from your two handwritten pages:

| File | Role |
|---|---|
| `quad_control_law.m` | The **Control Law** block — cascaded PID for altitude (T) + roll/pitch/yaw (Uy/Ux/Uz) |
| `quad_mixing_pwm.m`  | The **M⁻¹ + PWM** block — inverts the 4-motor mixing matrix, converts thrust → rotor speed → PWM |
| `build_quad_control_model.m` | Run this in MATLAB to auto-generate `quad_control_model.slx`, wiring the two blocks together exactly as in the diagram |

## Mapping from the photos

**Image 1 (block diagram + equations)**
```
[IMU, FP] --> [Control Law] --[Ux, Uy, T(via cos φ)]--> [M^-1] --> PWM
```
- `T = (m·g − Kp(z−zd) − Kd(ż−żd)) / cos φ` — I extended the denominator to `cos φ·cos θ`
  since thrust projects onto the world-vertical through *both* tilt axes, and added the
  integral term / saturation for a well-behaved discrete implementation.
- `Ux = −Kp(θ−θd) − Kd(θ̇−θ̇d) − Ki∫(θ−θd)` and `Uy = −Kp(φ−φd) − Kd(φ̇−φ̇d) − Ki∫(φ−φd)` —
  kept your exact labeling (Ux ↔ pitch θ, Uy ↔ roll φ). Swap the two at the call site if your
  airframe convention is the opposite.
- **Uz (yaw)** wasn't written out in the notes but is required by the mixing matrix in Image 2,
  so I added a standard PD yaw loop.

**Image 2 (mixing matrix)**
```
[T ]   [ 1    1    1    1 ] [M1]
[Ux] = [-cf   cf   cf  -cf] [M2]
[Uy]   [ cf   cf  -cf  -cf] [M3]
[Uz]   [-cd   cd  -cd   cd] [M4]
```
with `M_i = C_T·ω_i²` per rotor. `quad_mixing_pwm.m` inverts this matrix to solve for the 4
rotor thrusts from the commanded `[T,Ux,Uy,Uz]`, converts each to a rotor speed, then maps
speed → PWM (1000–2000 µs) with saturation.

## What I did **not** pull from the MATLAB script you attached
That script implements a **geometric SO(3) controller for a 3-rotor "bispinner"** — a different
vehicle and a different (more advanced, nonlinear) control strategy than the 4-rotor PID scheme
in the photos. Per your choice, this deliverable follows the photos/4-rotor path. I did reuse a
couple of physical constants from it (`m = 0.5 kg`, `g = 9.8`) since they're airframe-agnostic.
If you later want the geometric controller as a Simulink model too, that's a separate build
(different states, no Euler angles, SO(3) attitude error) — say the word and I'll do that one too.

## Tuning knobs you should revisit
All in `quad_control_law.m` (persistent gain block) and `quad_mixing_pwm.m` (persistent params
block): `Kp_z, Kd_z, Ki_z, Kp_att, Kd_att, Ki_att, Kp_yaw, Kd_yaw, T_max, U_max, cf, cd, CT,
omega_max, PWM_min/max`. None of these were specified precisely in your notes — I picked
plausible placeholder values and flagged them; they will need tuning against your real
airframe (mass, arm length, propeller/motor thrust coefficient).

## How to build the .slx
Requires MATLAB + Simulink (+ Stateflow, which ships with the MATLAB Function block):

```matlab
cd path/to/these/files
build_quad_control_model
```

This creates `quad_control_model.slx` with:
- Two vector inports: `State_IMU` (8: z,ż,φ,φ̇,θ,θ̇,ψ,ψ̇) and `Reference_FP` (8: same order, desired)
- `Control_Law` MATLAB Function block → outputs `[T,Ux,Uy,Uz]`
- `Mixing_PWM` MATLAB Function block → outputs `[PWM1..PWM4]`
- Scopes on both the torque/thrust commands and the PWM outputs
- Fixed-step discrete solver at `dt = 0.004 s` (matches the loop rate hardcoded in
  `quad_control_law.m` — keep these in sync if you change one)

I can't execute this script myself in this environment (no MATLAB here), so I haven't verified
it runs line-for-line — if a block name/path doesn't match your MATLAB version's library exactly,
that's the first thing to check.

## VYUHA-X Terrain Run game

The browser game and its optional Python bridges are in [`vyuha_game/`](./vyuha_game/). To run
the game in mouse/keyboard test mode, start a local web server from that directory:

```powershell
cd vyuha_game
python -m http.server 8000
```

Open `http://localhost:8000` in a browser and select a terrain. The game loads Three.js from a
CDN, so an internet connection is required. Optional Bluetooth/USB telemetry and webcam palm
tracking bridges require Python packages; see the usage text in `vyuha_game/vyuha.py` and
`vyuha_game/palm.py`. Run either bridge from `vyuha_game` when using its matching control mode.
