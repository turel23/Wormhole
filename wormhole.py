import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import matplotlib.pyplot as plt
#parameters
b_0 = 1
l = 6*b_0
l_max = 20*b_0
dt = 0.01
dt_max = 120
steps = int(dt_max/dt)
x_shift = -1270

# photon_l = np.zeros(steps)
# photon_theta = np.zeros(steps)
# photon_phi = np.zeros(steps)
# photon_dl = np.zeros(steps)
# photon_dtheta = np.zeros(steps)
# photon_dphi = np.zeros(steps)

#initial conditions
photon_l = l
photon_theta = np.pi/2
photon_phi = 0

#open  the hdri image
hdri_universe1 = imageio.imread("earthlike_planet.hdr")
hdri_universe2 = imageio.imread("HDR_galactic_plane_hazy_nebulae.hdr")
hdri_universe1 = hdri_universe1.astype(np.float32)
hdri_universe2 = hdri_universe2.astype(np.float32)
hdri_universe1 /= np.max(hdri_universe1)
hdri_universe2 /= np.max(hdri_universe2)


# plt.imshow(hdri_universe1)
# plt.title("Click to get coordinates")

# coords = plt.ginput(1)
# print(coords)
# exit()


H, W = 1024, 2048  # fake HDRI size

# # +l side: colorful gradient nebula
# hdri_universe1 = np.zeros((H, W, 3), dtype=np.float32)
# for i in range(H):
#     for j in range(W):
#         hdri_universe1[i,j] = [i/H, j/W, (i+j)/(H+W)]  # simple RGB gradient

# # -l side: dark / almost black
# hdri_universe2 = np.zeros((H, W, 3), dtype=np.float32)
# hdri_universe2 += 0.1  # slightly gray
#image setup
cam_resx = 200 #pixels
cam_resy = 200 
fov_y = 60 #degrees
fov_x = 60

image = np.zeros((cam_resy, cam_resx, 3), dtype=np.float32)
print(np.mean(hdri_universe1))
print(np.mean(hdri_universe2))

def accelerations(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi):
    #calculate the accelerations
    a_l = photon_l*(photon_dtheta**2 + np.sin(photon_theta)**2*photon_dphi**2)
    a_theta = np.sin(photon_theta)*np.cos(photon_theta)*photon_dphi**2 - 2*photon_l*photon_dl*photon_dtheta/(b_0**2+photon_l**2)
    # a_phi = -2*photon_l*photon_dl/(b_0**2+photon_l**2) - 2*photon_dtheta*photon_dphi/(np.tan(photon_theta)+1e-15)
    a_phi = -2*photon_l*photon_dl*photon_dphi/(b_0**2+photon_l**2) - 2*photon_dtheta*photon_dphi/(np.tan(photon_theta)+1e-15)
    return a_l, a_theta, a_phi
def derivatives(state):
    photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi = state
    a_l, a_theta, a_phi = accelerations(photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi)
    return np.array([photon_dl, photon_dtheta, photon_dphi, a_l, a_theta, a_phi])


capture = False
for i in range(cam_resx):
    if i % 20 == 0:
        print(f"Processing column {i}/{cam_resx}")
    for j in range(cam_resy):
        # photon_l = np.zeros(steps)
        # photon_theta = np.zeros(steps)
        # photon_phi = np.zeros(steps)
        # photon_dl = np.zeros(steps)
        # photon_dtheta = np.zeros(steps)
        # photon_dphi = np.zeros(steps)

        photon_l = l
        photon_theta = np.pi/2
        photon_phi = 0
        #calculate the angle of the pixel, in radians
        angle_x = ((i+0.5 - cam_resx/2) * fov_x / cam_resx)*np.pi/180
        angle_y = ((j+0.5 - cam_resy/2) * fov_y / cam_resy)*np.pi/180

        # intial conditions
        v_theta = -np.sin(angle_y)/np.sqrt(b_0**2+photon_l**2)
        v_phi = np.cos(angle_y)*np.sin(angle_x)/np.sqrt(b_0**2+photon_l**2)

        spatial_ang = (b_0**2 + photon_l**2)*(v_theta**2 + np.sin(photon_theta)**2*v_phi**2)
        v_l = -np.sqrt(1 - spatial_ang)
        
        photon_dl = v_l
        photon_dtheta = v_theta
        photon_dphi = v_phi
        
        state = np.array([photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi])
        last_index = steps-1
        for k in range(steps-1):
            k1 = derivatives(state)
            k2 = derivatives(state + 0.5*dt*k1)
            k3 = derivatives(state + 0.5*dt*k2)
            k4 = derivatives(state + dt*k3)
            state += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            photon_l, photon_theta, photon_phi, photon_dl, photon_dtheta, photon_dphi = state
            if abs(photon_l) > l_max:
                break
            if k == steps-1:
                capture = True
        if photon_l > 0:
            hdri = hdri_universe1
        else:
            hdri = hdri_universe2
            print(f"Ray {i},{j} crossed to negative side: l_final = {photon_l}")
        theta = photon_theta
        phi = photon_phi
        H, W = hdri.shape[:2]
        pixel_x = (int((phi / (2 * np.pi)) * W) + x_shift) % W
        pixel_y = (H - int((theta / np.pi) * H) - 1) % H

        color = hdri[pixel_y, pixel_x]
        if np.linalg.norm(color) < 0.01:
            color = np.array([0, 0, 0])  # treat very dark pixels as black
        image[j, i] = color
        if capture:
            image[j, i] = [0, 0, 0]  # captured rays appear black
        mid_x = cam_resx // 2
        mid_y = cam_resy // 2

        if abs(i - mid_x) < 10 and abs(j - mid_y) < 10:
            print(f"Pixel ({i},{j}) final l = {photon_l}")
        
#fix image
image = np.clip(image, 0, 1)
image = image**(1/2.2) #gamma correction
image_uint8 = (image*255).astype(np.uint8)
img = Image.fromarray(image_uint8)
img.save("simulation_success.png")
img.show()
