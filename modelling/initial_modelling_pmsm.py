import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Motor & system parameters ---
Rs = 0.5                # Stator resistance (Ω)
Ld = Lq = 0.001         # d/q inductances (H)

# For A2212 1000 Kv motor
Kv = 1000
Ke = (1 / Kv) * (2 * np.pi / 60)  # V per rad/s
p = 4                             # Pole-pairs
lambda_m = Ke / (1.5 * p)         # Flux linkage (Wb) ≈ 1.75e-5

J = 0.01              # Rotor inertia (kg·m²)
B = 1e-3              # Viscous friction
eta_rect = 0.9        # Efficiency
Vbatt = 9.0           # 9V battery
Cbatt = 2.0           # Smaller capacity for quick response
Rbat_int = 0.05       # Battery internal resistance

# --- Hub torque profile: represents input torque from gearbox side ---
def T_hub(t, omega):
    # Assume crank/hub applies enough torque to generate 0.5 Nm at generator side
    return 22 if t > 0 else 0.0

# --- PMSM & battery dynamics ---
def pmsm_dynamics(t, x):
    id_, iq, omega, qbatt = x
    vd = Rs * id_ - omega * Ld * iq
    vq = Rs * iq + omega * Ld * id_ + omega * lambda_m

    did_dt = (vd - Rs * id_ + omega * Lq * iq) / Ld
    diq_dt = (vq - Rs * iq - omega * Ld * id_) / Lq

    Te = 1.5 * p * lambda_m * iq
    domega_dt = (T_hub(t, omega) - Te - B * omega) / J

    Idc = 1.5 * iq
    Ibat = Idc * eta_rect
    dqbatt_dt = Ibat

    return [did_dt, diq_dt, domega_dt, dqbatt_dt]

# Initial state
x0 = [0, 0, 0, 0]

# Solve from t = 0 to 5 sec
sol = solve_ivp(pmsm_dynamics, [0, 5], x0, max_step=1e-3)

# Extract solution
t = sol.t
id_, iq, omega, qbatt = sol.y
Te = 1.5 * p * lambda_m * iq
Ibat = 1.5 * iq * eta_rect
rpm = omega * 60 / (2 * np.pi)

# --- Plots ---
plt.figure()
plt.plot(t, rpm, label='Speed (RPM)')
plt.xlabel('Time (s)')
plt.ylabel('RPM')
plt.legend()

plt.figure()
plt.plot(t, Te, label='Torque (Nm)')
plt.xlabel('Time (s)')
plt.ylabel('Torque')
plt.legend()

plt.figure()
plt.plot(t, Ibat, label='Battery Current (A)')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend()

plt.figure()
plt.plot(t, qbatt, label='Battery Charge (Ah)')
plt.xlabel('Time (s)')
plt.ylabel('Charge (Ah)')
plt.legend()

plt.show()
