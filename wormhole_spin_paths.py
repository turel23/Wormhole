import numpy as np
import imageio
from PIL import Image
from numba import cuda
import matplotlib.pyplot as plt
import math
import os
#parameters
b_0 = 1
l_max = 20*b_0
l = 6
dt = 0.002
a = -5 # Wormhole rotation parameter (0 for non-rotating, up to ~1 for fast rotation)
# Note for dt: base integration step used by curvature-adaptive stepping.
# With antialiasing and bilinear interpolation, dt=0.002-0.003 works well for 1080p.
# For 4K, use dt=0.001. Higher dt values may still work due to adaptive stepping.
dt_max = 120
steps = int(dt_max/dt)
rps = np.sqrt(-6*a)
K = 1
f_phi = 0
w = 2*a/(rps**3)
bc = rps*K/(np.exp(f_phi)+rps*K*w)
y = np.linspace(0, bc+2, 10)
x_init = 19
allray = []
energy_const = 1.0

# NOTE: photon_phi now starts at cam_phi+pi (to push the lensing fold caustic off-center).
# This shifts all phi lookups by W/2. To restore original background orientation,
# set x_shift = x_shift - hdri_universe1.shape[1]//2  (after loading the image).

#animation setting

def accelerations(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi, energy, momentum_phi):
    # Teo metric (N=1, K=1) in proper distance l with r^2 = l^2 + b0^2.
    r2 = b_0*b_0 + photon_l*photon_l
    sin_t = math.sin(photon_theta)
    cos_t = math.cos(photon_theta)
    sin2 = sin_t * sin_t
    omega = 2.0 * a * (r2 ** -1.5)
    domega_dl = -3.0 * photon_l * omega / r2

    gtt = -1.0 + r2 * sin2 * omega * omega
    gtphi = -r2 * sin2 * omega
    gphiphi = r2 * sin2

    delta = gtphi * gtphi - gtt * gphiphi
    if abs(delta) < 1e-12:
        delta = 1e-12

    photon_dt = (energy * gphiphi + momentum_phi * gtphi) / delta
    photon_dphi = -(energy * gtphi + momentum_phi * gtt) / delta

    psi = photon_dphi - omega * photon_dt
    a_l = photon_l * (photon_dtheta*photon_dtheta + sin2 * psi * psi) - r2 * sin2 * psi * domega_dl * photon_dt
    a_theta = ((photon_l**2+b_0**2)*sin_t*cos_t*(photon_dphi**2-2*omega*photon_dphi*photon_dt+omega*omega*photon_dt*photon_dt)-2*photon_l*photon_dl*photon_dtheta)/(photon_l**2+b_0**2)
    return photon_dt, photon_dphi, a_l, a_theta


def derivatives(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi, energy, momentum_phi):
    photon_dt, photon_dphi, a_l, a_theta = accelerations(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi, energy, momentum_phi)
    return photon_dt, photon_dphi, a_l, a_theta

def derived_dphi_from_conserved(photon_l, photon_theta, energy, momentum_phi):
    # Recompute dphi from conserved quantities at the updated state.
    r2 = b_0*b_0 + photon_l*photon_l
    sin_t = math.sin(photon_theta)
    sin2 = sin_t * sin_t
    omega = 2.0 * a * (r2 ** -1.5)

    gtt = -1.0 + r2 * sin2 * omega * omega
    gtphi = -r2 * sin2 * omega
    gphiphi = r2 * sin2

    delta = gtphi * gtphi - gtt * gphiphi
    if abs(delta) < 1e-12:
        delta = 1e-12

    return -(energy * gtphi + momentum_phi * gtt) / delta

def compute_adaptive_dt(base_dt, photon_l, photon_theta, photon_dtheta, photon_dphi):
    # Geometric curvature proxy is strongest near the throat and decays with distance.
    r2 = b_0*b_0 + photon_l*photon_l
    geom_curvature = (b_0*b_0) / r2

    sin_t = math.sin(photon_theta)
    ang_speed = r2 * (photon_dtheta*photon_dtheta + sin_t*sin_t*photon_dphi*photon_dphi)
    if ang_speed < 0.0:
        ang_speed = 0.0
    if ang_speed > 1.0:
        ang_speed = 1.0

    # Blend geometry and ray-direction dependence so each ray gets its own timestep.
    curvature_proxy = 0.75 * geom_curvature + 0.25 * ang_speed
    if curvature_proxy < 0.0:
        curvature_proxy = 0.0
    if curvature_proxy > 1.0:
        curvature_proxy = 1.0

    min_scale = 0.2
    max_scale = 1.8
    dt_scale = max_scale - (max_scale - min_scale) * curvature_proxy
    return base_dt * dt_scale

for y_init in np.concatenate([np.linspace(-bc-2, bc+2, 80), [bc-0.002485]]):
    path = []
    photon_l = np.sqrt(x_init**2+y_init**2-b_0**2)
    photon_phi = math.atan2(y_init, x_init)
    photon_theta = np.pi/2
    momentum_phi = y_init * energy_const
    
    # Use the Teo metric null condition to find initial dl/dt
    # For r >> b0, this is roughly: (dl/dt)^2 = E^2 - L^2/r^2
    r_init2 = photon_l**2 + b_0**2
    # We want it moving LEFT (negative x direction), so dl is negative
    photon_dl = -np.sqrt(max(0, energy_const**2 - (momentum_phi**2 / r_init2)))
    photon_dtheta = 0.0
    photon_dphi = derived_dphi_from_conserved(photon_l, photon_theta, energy_const, momentum_phi)
    for k in range(steps-1):
        photon_r = np.sqrt(photon_l**2 + b_0**2)
        path.append((photon_r, photon_phi))
        def get_k(l, theta, phi, dl, dtheta, dphi):
            dt_val, dphi_val, al, atheta = derivatives(l, theta, phi, dl, dtheta, dphi, energy_const, momentum_phi)
            return np.array([dl, dtheta, dphi_val, al, atheta])
        
        adaptive_dt = compute_adaptive_dt(dt, photon_l, photon_theta, photon_dtheta, photon_dphi)
        k1 = get_k(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi)
        k2 = get_k(photon_l + 0.5*adaptive_dt*k1[0], photon_theta + 0.5*adaptive_dt*k1[1], photon_phi + 0.5*adaptive_dt*k1[2], photon_dl + 0.5*adaptive_dt*k1[3], photon_dtheta + 0.5*adaptive_dt*k1[4], photon_dphi)
        k3 = get_k(photon_l + 0.5*adaptive_dt*k2[0], photon_theta + 0.5*adaptive_dt*k2[1], photon_phi + 0.5*adaptive_dt*k2[2], photon_dl + 0.5*adaptive_dt*k2[3], photon_dtheta + 0.5*adaptive_dt*k2[4], photon_dphi)
        k4 = get_k(photon_l + adaptive_dt*k3[0], photon_theta + adaptive_dt*k3[1], photon_phi + adaptive_dt*k3[2], photon_dl + adaptive_dt*k3[3], photon_dtheta + adaptive_dt*k3[4], photon_dphi)

        photon_l += (adaptive_dt/6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        photon_theta += (adaptive_dt/6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        photon_phi += (adaptive_dt/6) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        photon_dl += (adaptive_dt/6) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        photon_dtheta += (adaptive_dt/6) * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
        photon_dphi = derived_dphi_from_conserved(photon_l, photon_theta, energy_const, momentum_phi)
        # Check for numerical issues
        if not (-1e10 < photon_l < 1e10) or not (0 < photon_theta < math.pi):
            break
        # Stop tracing when photon approaches the wormhole throat
        if photon_l <= 0:
            break
    
    allray.append(path)

# ============================================
# Construct diagram of photon paths
# ============================================
fig, ax = plt.subplots(figsize=(12, 12))

# Plot each photon path (only those that don't pass through wormhole)
for path_idx, path in enumerate(allray):
    if len(path) > 0:
        # Convert (r, phi) to Cartesian (x, y)
        x_coords = [r * np.cos(phi) for r, phi in path]
        y_coords = [r * np.sin(phi) for r, phi in path]
        ax.plot(x_coords, y_coords, linewidth=1.5, alpha=0.7, color='blue')

# Mark the wormhole throat (r = b_0)
throat_circle = plt.Circle((0, 0), b_0, fill=False, color='red', linewidth=2.5, linestyle='-', label=f'Wormhole throat (r={b_0})')
ax.add_patch(throat_circle)

# Mark the retrograde photon sphere
r_ps = rps  # Retrograde orbital radius
ps_circle = plt.Circle((0, 0), r_ps, fill=False, color='orange', linewidth=3, linestyle='--', label=f'Retrograde photon sphere (r={r_ps:.3f})')
ax.add_patch(ps_circle)

# Mark starting points
for y_start in np.linspace(-bc-2, bc+2, 80):
    x_start = x_init
    ax.plot(x_start, y_start, 'g+', markersize=10, markeredgewidth=2)

ax.set_xlim(-bc-2-3, bc+2+3)
ax.set_ylim(-bc-2-3, bc+2+3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title(f'Retrograde Photon Paths on Equatorial Plane (a={a}, E=1.0)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('frames/photon_paths_diagram.png', dpi=150, bbox_inches='tight')
print(f"Diagram saved to frames/photon_paths_diagram.png ({len(allray)} rays plotted)")
plt.close()
