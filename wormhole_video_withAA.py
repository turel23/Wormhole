import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from numba import cuda
import math
import os

os.makedirs("frames", exist_ok=True)
#parameters
b_0 = 1
l_max = 20*b_0
dt = 0.002
#Note for dt: Adaptive timestep is now used near wormhole throat for better accuracy.
#With antialiasing and bilinear interpolation, dt=0.002-0.003 works well for 1080p.
#For 4K, use dt=0.001. Higher dt values may still work due to adaptive stepping.
dt_max = 120
steps = int(dt_max/dt)
aa_samples = 2  # antialiasing: number of samples per pixel (2 for diagonal sampling)
x_shift = 1500  # adjusted to move phi seam away from center

#animation settings
video_duration = 10
fps = 24
num_frames = video_duration * fps

change_phi = 0
phi_offset = 0
d_phi = change_phi/(num_frames-1) * (2*math.pi)/180 

change_l = -12
l = 6*b_0
d_l = change_l/(num_frames-1)

change_yaw = 180 #No need to convert to radians, it does it later in the kernel
cam_yaw = 0.0    # Rotate left/right
d_yaw = change_yaw/(num_frames-1)



#initial conditions
photon_l = l
photon_theta = np.pi/2
photon_phi = 0

#open  the hdri image
hdri_universe1 = imageio.imread("Skyboxes/starmap_2020_8K.exr")[..., :3].astype(np.float32)
hdri_universe2 = imageio.imread("Skyboxes/HDR_galactic_plane_hazy_nebulae.hdr")[..., :3].astype(np.float32)

print(f"Shape: {hdri_universe1.shape}, dtype: {hdri_universe1.dtype}") # troubleshooting

hdri_universe1 = hdri_universe1.astype(np.float32)
hdri_universe2 = hdri_universe2.astype(np.float32)
hdri_universe1 /= np.max(hdri_universe1)
hdri_universe2 /= np.max(hdri_universe2)



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



print(np.mean(hdri_universe1)) #troubleshooting
print(np.mean(hdri_universe2))


def fill_captured_with_horizontal_neighbors(image, captured_mask):
    """Replace captured pixels with left/right neighbor colors from same row."""
    if not np.any(captured_mask):
        return image

    fixed = image.copy()

    # Valid neighbor availability for each captured pixel.
    left_valid = np.zeros_like(captured_mask)
    right_valid = np.zeros_like(captured_mask)
    left_valid[:, 1:] = ~captured_mask[:, :-1]
    right_valid[:, :-1] = ~captured_mask[:, 1:]

    use_left = captured_mask & left_valid & ~right_valid
    use_right = captured_mask & right_valid & ~left_valid
    use_both = captured_mask & left_valid & right_valid

    # Fill from one side when only one clean neighbor exists.
    fixed[:, 1:][use_left[:, 1:]] = fixed[:, :-1][use_left[:, 1:]]
    fixed[:, :-1][use_right[:, :-1]] = fixed[:, 1:][use_right[:, :-1]]

    # Average left/right when both are available.
    both_cols = use_both[:, 1:-1]
    if np.any(both_cols):
        left_vals = fixed[:, :-2][both_cols]
        right_vals = fixed[:, 2:][both_cols]
        fixed[:, 1:-1][both_cols] = 0.5 * (left_vals + right_vals)

    return fixed

@cuda.jit(device=True)
def accelerations(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi):
    #calculate the accelerations
    inv_radius = 1.0 / (b_0*b_0 + photon_l*photon_l)
    sin_t = math.sin(photon_theta)
    cos_t = math.cos(photon_theta)
    if abs(sin_t) < 1e-8:
        sin_t = 1e-8  if sin_t >= 0 else -1e-8
    cot_t = cos_t / sin_t
    a_l = photon_l*(photon_dtheta*photon_dtheta + math.sin(photon_theta)*math.sin(photon_theta)*photon_dphi*photon_dphi)
    a_theta = math.sin(photon_theta)*math.cos(photon_theta)*photon_dphi*photon_dphi - 2*photon_l*photon_dl*photon_dtheta*inv_radius
    a_phi = -2*photon_l*photon_dl*photon_dphi*inv_radius - 2*photon_dtheta*photon_dphi*cot_t
    return a_l, a_theta, a_phi

@cuda.jit(device=True)
def derivatives(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi):
    a_l, a_theta, a_phi = accelerations(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi)
    return photon_dl, photon_dtheta, photon_dphi, a_l, a_theta, a_phi

@cuda.jit(fastmath = True)
def render_kernel(b_0, l, l_max, dt, steps, fov_x, fov_y, cam_resx, cam_resy, hdri_universe1_gpu, hdri_universe2_gpu, x_shift, image_gpu, capture_gpu, cam_phi, aa_samples, cam_pitch, cam_yaw):
    i, j = cuda.grid(2)
    if i >= cam_resx or j >= cam_resy:
        return

    # Antialiasing: accumulate color from multiple subpixel samples
    r_accum = 0.0
    g_accum = 0.0
    b_accum = 0.0
    
    # 2x2 sample grid for aa_samples=4
    samples_per_axis = int(math.sqrt(aa_samples))
    step_size = 1.0 / samples_per_axis
    
    for si in range(samples_per_axis):
        for sj in range(samples_per_axis):
            # Subpixel offset
            offset_x = (si + 0.5) * step_size
            offset_y = (sj + 0.5) * step_size
            
            photon_l = l
            photon_theta = math.pi/2
            photon_phi = cam_phi
            
            # Pinhole camera model: cast rays through a flat image plane.
            px = (2 * ((i + offset_x) / cam_resx) - 1) * math.tan(fov_x * math.pi / 360)
            py = (1 - 2 * ((j + offset_y) / cam_resy)) * math.tan(fov_y * math.pi / 360)

            # Add tiny epsilon to prevent exact zeros that might cause numerical issues
            epsilon = 1e-10
            if abs(px) < epsilon:
                px = epsilon if px >= 0 else -epsilon
            if abs(py) < epsilon:
                py = epsilon if py >= 0 else -epsilon

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

            spatial_ang = (b_0*b_0 + photon_l*photon_l)*(v_theta*v_theta + math.sin(photon_theta)*math.sin(photon_theta)*v_phi*v_phi)
            spatial_ang = min(spatial_ang, 1.0-1e-8)
            tmp = 1 - spatial_ang
            if tmp < 0:
                tmp = 0
            v_l = dir_z * math.sqrt(tmp)
            
            photon_dl = v_l
            photon_dtheta = v_theta
            photon_dphi = v_phi

            for k in range(steps-1):
                # Adaptive timestep: scale down near wormhole throat (dynamically updates as photon moves)
                radius_sq = b_0*b_0 + photon_l*photon_l
                adaptive_dt = dt
                
                if radius_sq < 4.0 * b_0 * b_0:  # Within 2*b_0 radius of throat
                    # Scale dt down smoothly as we get closer to throat
                    scale_factor = max(0.3, radius_sq / (4.0 * b_0 * b_0))
                    adaptive_dt = dt * scale_factor
                
                photon_dl_k1, photon_dtheta_k1, photon_dphi_k1, photon_al_k1, photon_atheta_k1, photon_aphi_k1 = derivatives(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi)
                photon_dl_k2, photon_dtheta_k2, photon_dphi_k2, photon_al_k2, photon_atheta_k2, photon_aphi_k2 = derivatives(photon_l + 0.5*adaptive_dt*photon_dl_k1, photon_theta + 0.5*adaptive_dt*photon_dtheta_k1, photon_phi + 0.5*adaptive_dt*photon_dphi_k1, photon_dl + 0.5*adaptive_dt*photon_al_k1, photon_dtheta + 0.5*adaptive_dt*photon_atheta_k1, photon_dphi + 0.5*adaptive_dt*photon_aphi_k1)
                photon_dl_k3, photon_dtheta_k3, photon_dphi_k3, photon_al_k3, photon_atheta_k3, photon_aphi_k3 = derivatives(photon_l + 0.5*adaptive_dt*photon_dl_k2, photon_theta + 0.5*adaptive_dt*photon_dtheta_k2, photon_phi + 0.5*adaptive_dt*photon_dphi_k2, photon_dl + 0.5*adaptive_dt*photon_al_k2, photon_dtheta + 0.5*adaptive_dt*photon_atheta_k2, photon_dphi + 0.5*adaptive_dt*photon_aphi_k2)
                photon_dl_k4, photon_dtheta_k4, photon_dphi_k4, photon_al_k4, photon_atheta_k4, photon_aphi_k4 = derivatives(photon_l + adaptive_dt*photon_dl_k3, photon_theta + adaptive_dt*photon_dtheta_k3, photon_phi + adaptive_dt*photon_dphi_k3, photon_dl + adaptive_dt*photon_al_k3, photon_dtheta + adaptive_dt*photon_atheta_k3, photon_dphi + adaptive_dt*photon_aphi_k3)
                photon_l += (adaptive_dt/6)*(photon_dl_k1 + 2*photon_dl_k2 + 2*photon_dl_k3 + photon_dl_k4)
                photon_theta += (adaptive_dt/6)*(photon_dtheta_k1 + 2*photon_dtheta_k2 + 2*photon_dtheta_k3 + photon_dtheta_k4)
                photon_phi += (adaptive_dt/6)*(photon_dphi_k1 + 2*photon_dphi_k2 + 2*photon_dphi_k3 + photon_dphi_k4)
                photon_dl += (adaptive_dt/6)*(photon_al_k1 + 2*photon_al_k2 + 2*photon_al_k3 + photon_al_k4)
                photon_dtheta += (adaptive_dt/6)*(photon_atheta_k1 + 2*photon_atheta_k2 + 2*photon_atheta_k3 + photon_atheta_k4)
                photon_dphi += (adaptive_dt/6)*(photon_aphi_k1 + 2*photon_aphi_k2 + 2*photon_aphi_k3 + photon_aphi_k4)
                
                # Check for numerical issues - mark pixel if this triggers
                if not (-1e10 < photon_l < 1e10) or not (0 < photon_theta < math.pi):
                    capture_gpu[j, i] = True
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
            # Accumulate samples
            r_accum += r
            g_accum += g
            b_accum += b
    
    # Average all samples
    image_gpu[j, i, 0] = r_accum / aa_samples
    image_gpu[j, i, 1] = g_accum / aa_samples
    image_gpu[j, i, 2] = b_accum / aa_samples
        

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
    cam_yaw_temp = 90 - 90 * (math.atan(l_temp / b_0) / math.atan(l / b_0))

    # Reset capture flags for this frame so only current-frame failures are repaired.
    capture.fill(False)
    capture_gpu.copy_to_device(capture)

    render_kernel[blockspergrid, threadsperblock](b_0, l_temp, l_max, dt, steps, fov_x, fov_y, cam_resx, cam_resy, hdri_universe1_gpu, hdri_universe2_gpu, x_shift, image_gpu, capture_gpu, phi_temp, aa_samples, cam_pitch, cam_yaw_temp)
    image = image_gpu.copy_to_host()
    capture = capture_gpu.copy_to_host()
    image = fill_captured_with_horizontal_neighbors(image, capture)
    image = np.clip(image, 0, 1)
    image = image**(1/2.2) #gamma correction
    image_uint8 = (image*255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    filename = f"frames/frame_{frame:04d}.png"
    img.save(filename)
    print(f"[PID {os.getpid()}] Frame {frame+1}/{num_frames} saved: {filename}", flush=True)