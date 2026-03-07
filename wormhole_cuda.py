import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from numba import cuda
import math

#parameters
b_0 = 1
l = 6*b_0
l_max = 20*b_0
dt = 0.003
dt_max = 120
steps = int(dt_max/dt)
x_shift = -1500  # adjusted to move phi seam away from center

#initial conditions
photon_l = l
photon_theta = np.pi/2
photon_phi = 0

#open  the hdri image
hdri_universe1 = imageio.imread("starmap_2020_8K.exr")[..., :3].astype(np.float32)
hdri_universe2 = imageio.imread("HDR_galactic_plane_hazy_nebulae.hdr")[..., :3].astype(np.float32)

print(f"Shape: {hdri_universe1.shape}, dtype: {hdri_universe1.dtype}")

hdri_universe1 = hdri_universe1.astype(np.float32)
hdri_universe2 = hdri_universe2.astype(np.float32)
hdri_universe1 /= np.max(hdri_universe1)
hdri_universe2 /= np.max(hdri_universe2)


# plt.imshow(hdri_universe1)
# plt.title("Click to get coordinates")

# coords = plt.ginput(1)
# print(coords)
# exit()
#image setup
cam_resx = 1920 #pixels
cam_resy = 1080 
fov_y = 60 #degrees
fov_x = 106.67

image = np.zeros((cam_resy, cam_resx, 3), dtype=np.float32)
image_gpu = cuda.to_device(image)
hdri_universe1_gpu = cuda.to_device(hdri_universe1)
hdri_universe2_gpu = cuda.to_device(hdri_universe2)
capture = np.zeros((cam_resy, cam_resx), dtype=np.bool_)
capture_gpu = cuda.to_device(capture)



print(np.mean(hdri_universe1))
print(np.mean(hdri_universe2))

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
def render_kernel(b_0, l, l_max, dt, steps, fov_x, fov_y, cam_resx, cam_resy, hdri_universe1_gpu, hdri_universe2_gpu, x_shift, image_gpu, capture_gpu):
    i, j = cuda.grid(2)
    if i >= cam_resx or j >= cam_resy:
        return

    photon_l = l
    photon_theta = math.pi/2
    photon_phi = 0
    #calculate the angle of the pixel, in radians
    angle_x = ((i+0.5 - cam_resx/2) * fov_x / cam_resx)*math.pi/180
    angle_y = ((j+0.5 - cam_resy/2) * fov_y / cam_resy)*math.pi/180+1e-8
    

    # intial conditions
    v_theta = -math.sin(angle_y)/math.sqrt(b_0*b_0+photon_l*photon_l)
    v_phi = math.cos(angle_y)*math.sin(angle_x)/math.sqrt(b_0*b_0+photon_l*photon_l)

    spatial_ang = (b_0*b_0 + photon_l*photon_l)*(v_theta*v_theta + math.sin(photon_theta)*math.sin(photon_theta)*v_phi*v_phi)
    spatial_ang = min(spatial_ang, 1.0-1e-8)
    tmp = 1 - spatial_ang
    if tmp < 0:
        tmp = 0
    v_l = -math.sqrt(tmp)
    
    photon_dl = v_l
    photon_dtheta = v_theta
    photon_dphi = v_phi

    for k in range(steps-1):
        photon_dl_k1, photon_dtheta_k1, photon_dphi_k1, photon_al_k1, photon_atheta_k1, photon_aphi_k1 = derivatives(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi)
        photon_dl_k2, photon_dtheta_k2, photon_dphi_k2, photon_al_k2, photon_atheta_k2, photon_aphi_k2 = derivatives(photon_l + 0.5*dt*photon_dl_k1, photon_theta + 0.5*dt*photon_dtheta_k1, photon_phi + 0.5*dt*photon_dphi_k1, photon_dl + 0.5*dt*photon_al_k1, photon_dtheta + 0.5*dt*photon_atheta_k1, photon_dphi + 0.5*dt*photon_aphi_k1)
        photon_dl_k3, photon_dtheta_k3, photon_dphi_k3, photon_al_k3, photon_atheta_k3, photon_aphi_k3 = derivatives(photon_l + 0.5*dt*photon_dl_k2, photon_theta + 0.5*dt*photon_dtheta_k2, photon_phi + 0.5*dt*photon_dphi_k2, photon_dl + 0.5*dt*photon_al_k2, photon_dtheta + 0.5*dt*photon_atheta_k2, photon_dphi + 0.5*dt*photon_aphi_k2)
        photon_dl_k4, photon_dtheta_k4, photon_dphi_k4, photon_al_k4, photon_atheta_k4, photon_aphi_k4 = derivatives(photon_l + dt*photon_dl_k3, photon_theta + dt*photon_dtheta_k3, photon_phi + dt*photon_dphi_k3, photon_dl + dt*photon_al_k3, photon_dtheta + dt*photon_atheta_k3, photon_dphi + dt*photon_aphi_k3)
        photon_l += (dt/6)*(photon_dl_k1 + 2*photon_dl_k2 + 2*photon_dl_k3 + photon_dl_k4)
        photon_theta += (dt/6)*(photon_dtheta_k1 + 2*photon_dtheta_k2 + 2*photon_dtheta_k3 + photon_dtheta_k4)
        photon_phi += (dt/6)*(photon_dphi_k1 + 2*photon_dphi_k2 + 2*photon_dphi_k3 + photon_dphi_k4)
        photon_dl += (dt/6)*(photon_al_k1 + 2*photon_al_k2 + 2*photon_al_k3 + photon_al_k4)
        photon_dtheta += (dt/6)*(photon_atheta_k1 + 2*photon_atheta_k2 + 2*photon_atheta_k3 + photon_atheta_k4)
        photon_dphi += (dt/6)*(photon_aphi_k1 + 2*photon_aphi_k2 + 2*photon_aphi_k3 + photon_aphi_k4)
        if abs(photon_l) > l_max:
            break
        # if k == steps-1:
        #     capture_gpu[j, i] = True
    
    theta = photon_theta
    phi = photon_phi
    phi = phi % (2 * math.pi) 
    # Clamp theta away from poles to avoid singularities
    theta = max(0.01, min(math.pi - 0.01, theta))
    H = hdri_universe1_gpu.shape[0]
    W = hdri_universe1_gpu.shape[1]
    pixel_x = (int((phi / (2 * math.pi)) * W) + x_shift) % W
    pixel_y = (H - int((theta / math.pi) * H) - 1) % H
    pixel_x = max(0, min(W - 1, pixel_x))
    pixel_y = max(0, min(H - 1, pixel_y))
    if photon_l > 0:
        r = hdri_universe1_gpu[pixel_y, pixel_x, 0]
        g = hdri_universe1_gpu[pixel_y, pixel_x, 1]
        b = hdri_universe1_gpu[pixel_y, pixel_x, 2]
    else:
        r = hdri_universe2_gpu[pixel_y, pixel_x, 0]
        g = hdri_universe2_gpu[pixel_y, pixel_x, 1]
        b = hdri_universe2_gpu[pixel_y, pixel_x, 2]

    norm = math.sqrt(r*r + g*g + b*b)
    # Temporarily disabled to debug - see raw HDRI values
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
    image_gpu[j, i, 0] = r
    image_gpu[j, i, 1] = g
    image_gpu[j, i, 2] = b
        

threadsperblock = (16,16)
blockspergrid_x = math.ceil(cam_resx / threadsperblock[0])
blockspergrid_y = math.ceil(cam_resy / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
render_kernel[blockspergrid, threadsperblock](b_0, l, l_max, dt, steps, fov_x, fov_y, cam_resx, cam_resy, hdri_universe1_gpu, hdri_universe2_gpu, x_shift, image_gpu, capture_gpu)
#fix image
image = image_gpu.copy_to_host()
image = np.clip(image, 0, 1)
image = image**(1/2.2) #gamma correction
image_uint8 = (image*255).astype(np.uint8)
img = Image.fromarray(image_uint8)
img.save("wormhole_1080p_test.png")
img.show()
