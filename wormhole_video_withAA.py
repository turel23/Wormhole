import numpy as np
import imageio
from PIL import Image
from numba import cuda
import matplotlib.pyplot as plt
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
cam_yaw = 40    # Rotate left/right
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
cam_resx = 1920 #pixels
cam_resy = 1080 
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
    a_theta = sin_t * cos_t * psi * psi - 2.0 * photon_l * photon_dl * photon_dtheta / r2
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
            
            
        

threadsperblock = (16,16)
blockspergrid_x = math.ceil(cam_resx / threadsperblock[0])
blockspergrid_y = math.ceil(cam_resy / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

print(f"Starting render: {num_frames} frames with {aa_samples}x antialiasing")
print(f"Process ID: {os.getpid()}")  # Help identify if multiple instances are running

