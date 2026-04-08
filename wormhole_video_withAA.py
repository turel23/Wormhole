import numpy as np
import imageio
from PIL import Image
from numba import cuda
import math
import os

os.makedirs("frames", exist_ok=True)
os.makedirs("videos", exist_ok=True)
#parameters
b_0 = 1
l_max = 20*b_0
dt = 0.002
a = 0.2  # Wormhole rotation parameter (0 for non-rotating, up to ~1 for fast rotation)
# Note for dt: base integration step used by curvature-adaptive stepping.
# With antialiasing and bilinear interpolation, dt=0.002-0.003 works well for 1080p.
# For 4K, use dt=0.001. Higher dt values may still work due to adaptive stepping.
dt_max = 120
steps = int(dt_max/dt)
aa_samples = 2  # antialiasing: number of samples per pixel (2 for diagonal sampling)
aa_samples = max(1, int(aa_samples))
x_shift = 1500  # base seam shift in pixels.
# Optional seam calibration for lensing-ring artifacts:
# 1) Temporarily set a = 0 and render a frame.
# 2) Measure the artifact longitude in degrees as seam_phi_deg.
# 3) Set seam_phi_deg below to map that longitude to u=0.
seam_phi_deg = None
# NOTE: photon_phi now starts at cam_phi+pi (to push the lensing fold caustic off-center).
# This shifts all phi lookups by W/2. To restore original background orientation,
# set x_shift = x_shift - hdri_universe1.shape[1]//2  (after loading the image).

#animation settings
video_duration = 1
fps = 1
num_frames = video_duration * fps
video_filename = "videos/wormhole_withAA.mp4"
frame_span = max(1, num_frames - 1)

change_phi = 0
phi_offset = 0
d_phi = change_phi/frame_span * (2*math.pi)/180 

change_l = -12
l = 6*b_0
d_l = change_l/frame_span

change_yaw = 180 #No need to convert to radians, it does it later in the kernel
cam_yaw = 0    # Rotate left/right
d_yaw = change_yaw/frame_span



#initial conditions
photon_l = l
photon_theta = np.pi/2
photon_phi = phi_offset * (2*math.pi)/180

#open  the hdri image
hdri_universe1 = imageio.imread("Skyboxes/starmap_2020_8k.exr")[..., :3].astype(np.float32)
hdri_universe2 = imageio.imread("Skyboxes/HDR_silver_and_gold_nebulae.hdr")[..., :3].astype(np.float32)

print(f"Shape: {hdri_universe1.shape}, dtype: {hdri_universe1.dtype}")

hdri_universe1 = hdri_universe1.astype(np.float32)
hdri_universe2 = hdri_universe2.astype(np.float32)
hdri_universe1 /= np.max(hdri_universe1)
hdri_universe2 /= np.max(hdri_universe2)

# Optional seam phase alignment so a known problematic phi lands on u=0.
if seam_phi_deg is not None:
    seam_align_px = int(round((-seam_phi_deg / 360.0) * hdri_universe1.shape[1]))
    x_shift += seam_align_px

# Compensate for the +pi photon_phi offset: shift x_shift by -W/2.
x_shift = (x_shift - hdri_universe1.shape[1] // 2) % hdri_universe1.shape[1]




#image setup
cam_resx = 1000 #pixels
cam_resy = 1000 
fov_y = 60 #degrees
aspect = cam_resx / cam_resy
# Derive fov_x from fov_y and aspect to avoid squeeze
fov_x = 2 * math.atan(math.tan(fov_y * math.pi / 360) * aspect) * 180 / math.pi
# Camera rotation angles in degrees (pitch up toward north pole, yaw)
cam_pitch = 0  # Increase to look UP (e.g., 45.0 to look at north pole)


image = np.zeros((cam_resy, cam_resx, 3), dtype=np.float32)
image_gpu = cuda.to_device(image)
hdri_universe1_gpu = cuda.to_device(hdri_universe1)
hdri_universe2_gpu = cuda.to_device(hdri_universe2)
capture = np.zeros((cam_resy, cam_resx), dtype=np.bool_)
capture_gpu = cuda.to_device(capture)



print(np.mean(hdri_universe1))
print(np.mean(hdri_universe2))

@cuda.jit(device=True)
def initial_conditions(photon_l, photon_theta, photon_dt, photon_dphi):
    r2 = b_0*b_0 + photon_l*photon_l
    sin_t = math.sin(photon_theta)
    sin2 = sin_t * sin_t
    omega = 2.0 * a * (r2 ** -1.5)

    gtt = -1.0 + r2 * sin2 * omega * omega
    gtphi = -r2 * sin2 * omega
    gphiphi = r2 * sin2

    energy = -(gtt * photon_dt + gtphi * photon_dphi)
    momentum = gtphi * photon_dt + gphiphi * photon_dphi
    return energy, momentum

@cuda.jit(device=True)
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

@cuda.jit(device=True)
def derivatives(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi, energy, momentum_phi):
    photon_dt, photon_dphi, a_l, a_theta = accelerations(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi, energy, momentum_phi)
    return photon_dt, photon_dphi, a_l, a_theta

@cuda.jit(device=True)
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

@cuda.jit(device=True)
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

@cuda.jit(fastmath = True)
def render_kernel(b_0, l, l_max, dt, steps, fov_x, fov_y, cam_resx, cam_resy, hdri_universe1_gpu, hdri_universe2_gpu, x_shift, image_gpu, capture_gpu, cam_phi, aa_samples, cam_pitch, cam_yaw, frame_seed):
    i, j = cuda.grid(2)
    if i >= cam_resx or j >= cam_resy:
        return

    # Antialiasing: accumulate color from multiple subpixel samples
    r_accum = 0.0
    g_accum = 0.0
    b_accum = 0.0
    
    sample_count = 0

    # Explicitly support aa_samples=2 (diagonal pattern) and square-grid AA for other counts.
    if aa_samples == 2:
        for s in range(2):
            if s == 0:
                offset_x = 0.25
                offset_y = 0.25
            else:
                offset_x = 0.75
                offset_y = 0.75
            sample_count += 1
            
            photon_l = l
            photon_theta = math.pi/2
            photon_phi = cam_phi + math.pi

            # Pinhole camera model: cast rays through a flat image plane.
            px = (2 * ((i + offset_x) / cam_resx) - 1) * math.tan(fov_x * math.pi / 360)
            py = (1 - 2 * ((j + offset_y) / cam_resy)) * math.tan(fov_y * math.pi / 360)

            dir_x = px
            dir_y = py
            dir_z = -1.0

            # Apply camera pitch and yaw rotations.
            # Clamp away from exact +/-90 deg where longitude is undefined at poles.
            pitch_rad = cam_pitch * math.pi / 180.0
            pitch_limit = 89.5 * math.pi / 180.0
            if pitch_rad > pitch_limit:
                pitch_rad = pitch_limit
            elif pitch_rad < -pitch_limit:
                pitch_rad = -pitch_limit
            yaw_rad = cam_yaw * math.pi / 180.0

            # Pitch (rotate around X-axis): positive pitch looks upward at poles.
            dir_y_p = dir_y * math.cos(pitch_rad) - dir_z * math.sin(pitch_rad)
            dir_z_p = dir_y * math.sin(pitch_rad) + dir_z * math.cos(pitch_rad)

            # Yaw (rotate around Y-axis): positive yaw looks left.
            dir_x_y = dir_x * math.cos(yaw_rad) - dir_z_p * math.sin(yaw_rad)
            dir_z_f = dir_x * math.sin(yaw_rad) + dir_z_p * math.cos(yaw_rad)

            dir_x = dir_x_y
            dir_y = dir_y_p
            dir_z = dir_z_f

            dir_norm = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
            dir_x /= dir_norm
            dir_y /= dir_norm
            dir_z /= dir_norm

            # Convert 3D direction to spherical velocity components.
            # Camera starts at equator (theta=π/2) looking in +phi direction.
            # dir_z is forward (along +phi), dir_x is right, dir_y is up (toward pole).
            inv_r = 1.0 / math.sqrt(b_0 * b_0 + photon_l * photon_l)
            # Tangent-plane mapping at camera location:
            # +x on sensor -> +phi, +y on sensor -> -theta, z contributes to radial part (v_l).
            v_theta = -dir_y * inv_r
            v_phi = dir_x * inv_r

            # v_l: null constraint gives v_l^2 = 1 - r^2*(v_theta^2 + v_phi^2) = dir_z^2,
            # so v_l = dir_z (preserving sign). The old dir_z*sqrt(tmp) computed dir_z*|dir_z|
            # which has wrong magnitude for oblique rays.
            v_l = dir_z

            photon_dl = v_l
            photon_dtheta = v_theta
            photon_dphi = v_phi
            photon_dt = 1.0

            energy, momentum = initial_conditions(photon_l, photon_theta, photon_dt, photon_dphi)

            for k in range(steps-1):
                adaptive_dt = compute_adaptive_dt(dt, photon_l, photon_theta, photon_dtheta, photon_dphi)

                k1_dt, k1_dphi, k1_al, k1_atheta = derivatives(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi, energy, momentum)
                k1_l = photon_dl
                k1_theta = photon_dtheta
                k1_phi = k1_dphi
                k1_dl = k1_al
                k1_dtheta = k1_atheta

                l2 = photon_l + 0.5 * adaptive_dt * k1_l
                theta2 = photon_theta + 0.5 * adaptive_dt * k1_theta
                phi2 = photon_phi + 0.5 * adaptive_dt * k1_phi
                dl2 = photon_dl + 0.5 * adaptive_dt * k1_dl
                dtheta2 = photon_dtheta + 0.5 * adaptive_dt * k1_dtheta
                dphi2 = k1_dphi
                k2_dt, k2_dphi, k2_al, k2_atheta = derivatives(l2, theta2, phi2, dl2, dtheta2, dphi2, energy, momentum)
                k2_l = dl2
                k2_theta = dtheta2
                k2_phi = k2_dphi
                k2_dl = k2_al
                k2_dtheta = k2_atheta

                l3 = photon_l + 0.5 * adaptive_dt * k2_l
                theta3 = photon_theta + 0.5 * adaptive_dt * k2_theta
                phi3 = photon_phi + 0.5 * adaptive_dt * k2_phi
                dl3 = photon_dl + 0.5 * adaptive_dt * k2_dl
                dtheta3 = photon_dtheta + 0.5 * adaptive_dt * k2_dtheta
                dphi3 = k2_dphi
                k3_dt, k3_dphi, k3_al, k3_atheta = derivatives(l3, theta3, phi3, dl3, dtheta3, dphi3, energy, momentum)
                k3_l = dl3
                k3_theta = dtheta3
                k3_phi = k3_dphi
                k3_dl = k3_al
                k3_dtheta = k3_atheta

                l4 = photon_l + adaptive_dt * k3_l
                theta4 = photon_theta + adaptive_dt * k3_theta
                phi4 = photon_phi + adaptive_dt * k3_phi
                dl4 = photon_dl + adaptive_dt * k3_dl
                dtheta4 = photon_dtheta + adaptive_dt * k3_dtheta
                dphi4 = k3_dphi
                k4_dt, k4_dphi, k4_al, k4_atheta = derivatives(l4, theta4, phi4, dl4, dtheta4, dphi4, energy, momentum)
                k4_l = dl4
                k4_theta = dtheta4
                k4_phi = k4_dphi
                k4_dl = k4_al
                k4_dtheta = k4_atheta

                photon_l += (adaptive_dt/6) * (k1_l + 2*k2_l + 2*k3_l + k4_l)
                photon_theta += (adaptive_dt/6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
                photon_phi += (adaptive_dt/6) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
                photon_dl += (adaptive_dt/6) * (k1_dl + 2*k2_dl + 2*k3_dl + k4_dl)
                photon_dtheta += (adaptive_dt/6) * (k1_dtheta + 2*k2_dtheta + 2*k3_dtheta + k4_dtheta)
                photon_dphi = derived_dphi_from_conserved(photon_l, photon_theta, energy, momentum)
                
                # Check for numerical issues
                if not (-1e10 < photon_l < 1e10) or not (0 < photon_theta < math.pi):
                    break
                if abs(photon_l) > l_max:
                    break
            
            # Equirectangular environment lookup with explicit pole-zone stabilization.
            phi = photon_phi % (2 * math.pi)
            theta = max(0.0, min(math.pi, photon_theta))

            H = hdri_universe1_gpu.shape[0]
            W = hdri_universe1_gpu.shape[1]

            # Use a practical pole zone (~2 texel rows) where longitude becomes unstable.
            pole_eps = 2.0 * (math.pi / H)

            # Horizontal coordinate from longitude.
            u = (phi / (2 * math.pi)) * W
            # Near poles, force a stable longitude to avoid streaking/tearing.
            if theta < pole_eps or theta > (math.pi - pole_eps):
                u = 0.5 * W

            # Vertical coordinate from latitude (standard equirectangular map).
            v = (theta / math.pi) * H

            # Apply seam shift after computing longitude.
            u = (u + x_shift) % W

            # Get integer coordinates.
            u0 = int(u) % W
            v0 = int(v)
            u1 = (u0 + 1) % W
            v1 = min(v0 + 1, H - 1)
            v0 = max(0, min(H - 1, v0))

            # Bilinear interpolation weights.
            fu = u - int(u)
            fv = v - int(v)
            
            if photon_l > 0:
                # Bilinear interpolation
                c00_r = hdri_universe1_gpu[v0, u0, 0]
                c00_g = hdri_universe1_gpu[v0, u0, 1]
                c00_b = hdri_universe1_gpu[v0, u0, 2]
                c01_r = hdri_universe1_gpu[v0, u1, 0]
                c01_g = hdri_universe1_gpu[v0, u1, 1]
                c01_b = hdri_universe1_gpu[v0, u1, 2]
                c10_r = hdri_universe1_gpu[v1, u0, 0]
                c10_g = hdri_universe1_gpu[v1, u0, 1]
                c10_b = hdri_universe1_gpu[v1, u0, 2]
                c11_r = hdri_universe1_gpu[v1, u1, 0]
                c11_g = hdri_universe1_gpu[v1, u1, 1]
                c11_b = hdri_universe1_gpu[v1, u1, 2]
                
                r = (1-fu)*(1-fv)*c00_r + fu*(1-fv)*c01_r + (1-fu)*fv*c10_r + fu*fv*c11_r
                g = (1-fu)*(1-fv)*c00_g + fu*(1-fv)*c01_g + (1-fu)*fv*c10_g + fu*fv*c11_g
                b = (1-fu)*(1-fv)*c00_b + fu*(1-fv)*c01_b + (1-fu)*fv*c10_b + fu*fv*c11_b
            else:
                # Bilinear interpolation
                c00_r = hdri_universe2_gpu[v0, u0, 0]
                c00_g = hdri_universe2_gpu[v0, u0, 1]
                c00_b = hdri_universe2_gpu[v0, u0, 2]
                c01_r = hdri_universe2_gpu[v0, u1, 0]
                c01_g = hdri_universe2_gpu[v0, u1, 1]
                c01_b = hdri_universe2_gpu[v0, u1, 2]
                c10_r = hdri_universe2_gpu[v1, u0, 0]
                c10_g = hdri_universe2_gpu[v1, u0, 1]
                c10_b = hdri_universe2_gpu[v1, u0, 2]
                c11_r = hdri_universe2_gpu[v1, u1, 0]
                c11_g = hdri_universe2_gpu[v1, u1, 1]
                c11_b = hdri_universe2_gpu[v1, u1, 2]
                
                r = (1-fu)*(1-fv)*c00_r + fu*(1-fv)*c01_r + (1-fu)*fv*c10_r + fu*fv*c11_r
                g = (1-fu)*(1-fv)*c00_g + fu*(1-fv)*c01_g + (1-fu)*fv*c10_g + fu*fv*c11_g
                b = (1-fu)*(1-fv)*c00_b + fu*(1-fv)*c01_b + (1-fu)*fv*c10_b + fu*fv*c11_b

            norm = math.sqrt(r*r + g*g + b*b)
            if norm < 0.05:
                # Desaturate and fade very dark pixels
                gray = (r + g + b) / 3.0
                fade = norm / 0.05
                r = gray * fade * 0.5
                g = gray * fade * 0.5
                b = gray * fade * 0.5
            elif norm > 0:
                # Smooth fade below 0.1
                fade = min(1.0, norm / 0.1)
                r *= fade
                g *= fade
                b *= fade
            if capture_gpu[j, i]:
                r, g, b = 0, 0, 0  # captured rays appear black
            
            # Accumulate samples
            r_accum += r
            g_accum += g
            b_accum += b
    else:
        samples_per_axis = int(math.ceil(math.sqrt(aa_samples)))
        if samples_per_axis < 1:
            samples_per_axis = 1
        step_size = 1.0 / samples_per_axis

        for si in range(samples_per_axis):
            for sj in range(samples_per_axis):
                if sample_count >= aa_samples:
                    continue
                # Stratified jitter to avoid coherent AA grid artifacts on caustics.
                seed = (i * 1973 + j * 9277 + si * 13 + sj * 7 + frame_seed * 26699) & 0xFFFFFF
                rand_x = (seed & 0xFFF) / 4095.0
                rand_y = ((seed >> 12) & 0xFFF) / 4095.0
                offset_x = (si + rand_x) * step_size
                offset_y = (sj + rand_y) * step_size
                sample_count += 1

                photon_l = l
                photon_theta = math.pi/2
                photon_phi = cam_phi + math.pi

                # Pinhole camera model: cast rays through a flat image plane.
                px = (2 * ((i + offset_x) / cam_resx) - 1) * math.tan(fov_x * math.pi / 360)
                py = (1 - 2 * ((j + offset_y) / cam_resy)) * math.tan(fov_y * math.pi / 360)

                dir_x = px
                dir_y = py
                dir_z = -1.0

                # Apply camera pitch and yaw rotations.
                # Clamp away from exact +/-90 deg where longitude is undefined at poles.
                pitch_rad = cam_pitch * math.pi / 180.0
                pitch_limit = 89.5 * math.pi / 180.0
                if pitch_rad > pitch_limit:
                    pitch_rad = pitch_limit
                elif pitch_rad < -pitch_limit:
                    pitch_rad = -pitch_limit
                yaw_rad = cam_yaw * math.pi / 180.0

                # Pitch (rotate around X-axis): positive pitch looks upward at poles.
                dir_y_p = dir_y * math.cos(pitch_rad) - dir_z * math.sin(pitch_rad)
                dir_z_p = dir_y * math.sin(pitch_rad) + dir_z * math.cos(pitch_rad)

                # Yaw (rotate around Y-axis): positive yaw looks left.
                dir_x_y = dir_x * math.cos(yaw_rad) - dir_z_p * math.sin(yaw_rad)
                dir_z_f = dir_x * math.sin(yaw_rad) + dir_z_p * math.cos(yaw_rad)

                dir_x = dir_x_y
                dir_y = dir_y_p
                dir_z = dir_z_f

                dir_norm = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
                dir_x /= dir_norm
                dir_y /= dir_norm
                dir_z /= dir_norm

                # Convert 3D direction to spherical velocity components.
                # Camera starts at equator (theta=π/2) looking in +phi direction.
                # dir_z is forward (along +phi), dir_x is right, dir_y is up (toward pole).
                inv_r = 1.0 / math.sqrt(b_0 * b_0 + photon_l * photon_l)
                # Tangent-plane mapping at camera location:
                # +x on sensor -> +phi, +y on sensor -> -theta, z contributes to radial part (v_l).
                v_theta = -dir_y * inv_r
                v_phi = dir_x * inv_r

                # v_l: null constraint gives v_l = dir_z (see aa_samples==2 branch for derivation).
                v_l = dir_z

                photon_dl = v_l
                photon_dtheta = v_theta
                photon_dphi = v_phi
                photon_dt = 1.0

                energy, momentum = initial_conditions(photon_l, photon_theta, photon_dt, photon_dphi)
                for k in range(steps-1):
                    adaptive_dt = compute_adaptive_dt(dt, photon_l, photon_theta, photon_dtheta, photon_dphi)

                    k1_dt, k1_dphi, k1_al, k1_atheta = derivatives(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi, energy, momentum)
                    k1_l = photon_dl
                    k1_theta = photon_dtheta
                    k1_phi = k1_dphi
                    k1_dl = k1_al
                    k1_dtheta = k1_atheta

                    l2 = photon_l + 0.5 * adaptive_dt * k1_l
                    theta2 = photon_theta + 0.5 * adaptive_dt * k1_theta
                    phi2 = photon_phi + 0.5 * adaptive_dt * k1_phi
                    dl2 = photon_dl + 0.5 * adaptive_dt * k1_dl
                    dtheta2 = photon_dtheta + 0.5 * adaptive_dt * k1_dtheta
                    dphi2 = k1_dphi
                    k2_dt, k2_dphi, k2_al, k2_atheta = derivatives(l2, theta2, phi2, dl2, dtheta2, dphi2, energy, momentum)
                    k2_l = dl2
                    k2_theta = dtheta2
                    k2_phi = k2_dphi
                    k2_dl = k2_al
                    k2_dtheta = k2_atheta

                    l3 = photon_l + 0.5 * adaptive_dt * k2_l
                    theta3 = photon_theta + 0.5 * adaptive_dt * k2_theta
                    phi3 = photon_phi + 0.5 * adaptive_dt * k2_phi
                    dl3 = photon_dl + 0.5 * adaptive_dt * k2_dl
                    dtheta3 = photon_dtheta + 0.5 * adaptive_dt * k2_dtheta
                    dphi3 = k2_dphi
                    k3_dt, k3_dphi, k3_al, k3_atheta = derivatives(l3, theta3, phi3, dl3, dtheta3, dphi3, energy, momentum)
                    k3_l = dl3
                    k3_theta = dtheta3
                    k3_phi = k3_dphi
                    k3_dl = k3_al
                    k3_dtheta = k3_atheta

                    l4 = photon_l + adaptive_dt * k3_l
                    theta4 = photon_theta + adaptive_dt * k3_theta
                    phi4 = photon_phi + adaptive_dt * k3_phi
                    dl4 = photon_dl + adaptive_dt * k3_dl
                    dtheta4 = photon_dtheta + adaptive_dt * k3_dtheta
                    dphi4 = k3_dphi
                    k4_dt, k4_dphi, k4_al, k4_atheta = derivatives(l4, theta4, phi4, dl4, dtheta4, dphi4, energy, momentum)
                    k4_l = dl4
                    k4_theta = dtheta4
                    k4_phi = k4_dphi
                    k4_dl = k4_al
                    k4_dtheta = k4_atheta

                    photon_l += (adaptive_dt/6) * (k1_l + 2*k2_l + 2*k3_l + k4_l)
                    photon_theta += (adaptive_dt/6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
                    photon_phi += (adaptive_dt/6) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
                    photon_dl += (adaptive_dt/6) * (k1_dl + 2*k2_dl + 2*k3_dl + k4_dl)
                    photon_dtheta += (adaptive_dt/6) * (k1_dtheta + 2*k2_dtheta + 2*k3_dtheta + k4_dtheta)
                    photon_dphi = derived_dphi_from_conserved(photon_l, photon_theta, energy, momentum)
                    
                    # Check for numerical issues
                    if not (-1e10 < photon_l < 1e10) or not (0 < photon_theta < math.pi):
                        break
                    if abs(photon_l) > l_max:
                        break
                
                # Equirectangular environment lookup with explicit pole-zone stabilization.
                phi = photon_phi % (2 * math.pi)
                theta = max(0.0, min(math.pi, photon_theta))

                H = hdri_universe1_gpu.shape[0]
                W = hdri_universe1_gpu.shape[1]

                # Use a practical pole zone (~2 texel rows) where longitude becomes unstable.
                pole_eps = 2.0 * (math.pi / H)

                # Horizontal coordinate from longitude.
                u = (phi / (2 * math.pi)) * W
                # Near poles, force a stable longitude to avoid streaking/tearing.
                if theta < pole_eps or theta > (math.pi - pole_eps):
                    u = 0.5 * W

                # Vertical coordinate from latitude (standard equirectangular map).
                v = (theta / math.pi) * H

                # Apply seam shift after computing longitude.
                u = (u + x_shift) % W

                # Get integer coordinates.
                u0 = int(u) % W
                v0 = int(v)
                u1 = (u0 + 1) % W
                v1 = min(v0 + 1, H - 1)
                v0 = max(0, min(H - 1, v0))

                # Bilinear interpolation weights.
                fu = u - int(u)
                fv = v - int(v)
                
                if photon_l > 0:
                    # Bilinear interpolation
                    c00_r = hdri_universe1_gpu[v0, u0, 0]
                    c00_g = hdri_universe1_gpu[v0, u0, 1]
                    c00_b = hdri_universe1_gpu[v0, u0, 2]
                    c01_r = hdri_universe1_gpu[v0, u1, 0]
                    c01_g = hdri_universe1_gpu[v0, u1, 1]
                    c01_b = hdri_universe1_gpu[v0, u1, 2]
                    c10_r = hdri_universe1_gpu[v1, u0, 0]
                    c10_g = hdri_universe1_gpu[v1, u0, 1]
                    c10_b = hdri_universe1_gpu[v1, u0, 2]
                    c11_r = hdri_universe1_gpu[v1, u1, 0]
                    c11_g = hdri_universe1_gpu[v1, u1, 1]
                    c11_b = hdri_universe1_gpu[v1, u1, 2]
                    
                    r = (1-fu)*(1-fv)*c00_r + fu*(1-fv)*c01_r + (1-fu)*fv*c10_r + fu*fv*c11_r
                    g = (1-fu)*(1-fv)*c00_g + fu*(1-fv)*c01_g + (1-fu)*fv*c10_g + fu*fv*c11_g
                    b = (1-fu)*(1-fv)*c00_b + fu*(1-fv)*c01_b + (1-fu)*fv*c10_b + fu*fv*c11_b
                else:
                    # Bilinear interpolation
                    c00_r = hdri_universe2_gpu[v0, u0, 0]
                    c00_g = hdri_universe2_gpu[v0, u0, 1]
                    c00_b = hdri_universe2_gpu[v0, u0, 2]
                    c01_r = hdri_universe2_gpu[v0, u1, 0]
                    c01_g = hdri_universe2_gpu[v0, u1, 1]
                    c01_b = hdri_universe2_gpu[v0, u1, 2]
                    c10_r = hdri_universe2_gpu[v1, u0, 0]
                    c10_g = hdri_universe2_gpu[v1, u0, 1]
                    c10_b = hdri_universe2_gpu[v1, u0, 2]
                    c11_r = hdri_universe2_gpu[v1, u1, 0]
                    c11_g = hdri_universe2_gpu[v1, u1, 1]
                    c11_b = hdri_universe2_gpu[v1, u1, 2]
                    
                    r = (1-fu)*(1-fv)*c00_r + fu*(1-fv)*c01_r + (1-fu)*fv*c10_r + fu*fv*c11_r
                    g = (1-fu)*(1-fv)*c00_g + fu*(1-fv)*c01_g + (1-fu)*fv*c10_g + fu*fv*c11_g
                    b = (1-fu)*(1-fv)*c00_b + fu*(1-fv)*c01_b + (1-fu)*fv*c10_b + fu*fv*c11_b

                norm = math.sqrt(r*r + g*g + b*b)
                if norm < 0.05:
                    # Desaturate and fade very dark pixels
                    gray = (r + g + b) / 3.0
                    fade = norm / 0.05
                    r = gray * fade * 0.5
                    g = gray * fade * 0.5
                    b = gray * fade * 0.5
                elif norm > 0:
                    # Smooth fade below 0.1
                    fade = min(1.0, norm / 0.1)
                    r *= fade
                    g *= fade
                    b *= fade
                if capture_gpu[j, i]:
                    r, g, b = 0, 0, 0  # captured rays appear black
                
                # Accumulate samples
                r_accum += r
                g_accum += g
                b_accum += b

    if sample_count < 1:
        sample_count = 1
    
    # Average all samples
    image_gpu[j, i, 0] = r_accum / sample_count
    image_gpu[j, i, 1] = g_accum / sample_count
    image_gpu[j, i, 2] = b_accum / sample_count
        

threadsperblock = (16,16)
blockspergrid_x = math.ceil(cam_resx / threadsperblock[0])
blockspergrid_y = math.ceil(cam_resy / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

print(f"Starting render: {num_frames} frames with {aa_samples}x antialiasing")
print(f"Process ID: {os.getpid()}")  # Help identify if multiple instances are running

for frame in range(num_frames):
    phi_temp = phi_offset + (frame) * d_phi
    l_temp = l + (frame) * d_l
    # Non-linear yaw using atan to keep wormhole centered while rotating
    cam_yaw_temp =cam_yaw+ 90 - 90 * (math.atan(l_temp / b_0) / math.atan(l / b_0))
    render_kernel[blockspergrid, threadsperblock](b_0, l_temp, l_max, dt, steps, fov_x, fov_y, cam_resx, cam_resy, hdri_universe1_gpu, hdri_universe2_gpu, x_shift, image_gpu, capture_gpu, phi_temp, aa_samples, cam_pitch, cam_yaw_temp, frame)
    image = image_gpu.copy_to_host()
    image = np.clip(image, 0, 1)
    image = image**(1/2.2) #gamma correction
    image_uint8 = (image*255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    filename = f"frames/frame_{frame:04d}.png"
    img.save(filename)
    print(f"[PID {os.getpid()}] Frame {frame+1}/{num_frames} saved: {filename}", flush=True)

print(f"Assembling video: {video_filename}")
try:
    with imageio.get_writer(video_filename, fps=fps) as writer:
        for frame in range(num_frames):
            frame_path = f"frames/frame_{frame:04d}.png"
            writer.append_data(imageio.imread(frame_path))
    print(f"Video saved: {video_filename}")
except Exception as exc:
    print(f"Video assembly failed ({exc}). Frames are still available in ./frames")