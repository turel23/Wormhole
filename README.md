# Wormhole
This project uses relativstic ray tracing to create a visual simulation for wormholes

I used the metric for the Ellis wormhole to derive the equations that I integrated with standard Runge-Kutta 4
The files for wormhole.py and wormhole_cuda.py were written mostly if not entirely by myself, though for more complicated camera work and fixing stretching (which was a problem because it projeted a 2d image into a 3d space) I had implemented with Claude Haiku 4.5 and GPT 5.3 Codex. 

# Structure
The intention of this project was to make it easily accessible to anyone who wants to create their own simulations of a wormhole and change the parameters to their own desire, or to rewrite it completely for some other setup. 
The structure of the project is set up so that there are different versions of the same basic code that serve different purposes. Because I have an Nvidia RTX 5070ti, I used Numba Cuda to run the parallel computations on my graphics card. The first file, wormhole.py, runs the computations on CPU instead. Rendering an image on my CPU took about 40 minutes.

wormhole_cuda.py uses Numba Cuda which reduced my rendering time significantly. Rendering a 200X200 image on cpu took 40 minutes and on gpu it took less than a second. Then, wormhole_video.py builds on the Cuda version to create a video, though it does not have any features that improves the quality of the videos.

wormhole_video_withAA.py includes more features. Since it has anti-aliasing, it is _withAA, and I have it set to 2 as default, so it takes 2 pixels instead of just 1. It also has bilinear interpolation and a better camera model, and fixes the weird distortion at the poles. Running 4x antialiasing means it will take 4 times as long compared to rendering a normal frame without anti-aliasing.

# Problems
Some problems I ran into included artifacts that showed up in the middle of the screen. For higher dt values, which is the step size, a vertical column would appear covering the wormhole. For higher resolutions, it showed up even more, which meant I had to decrease the step size for higher resolutions. I didn't want to increase the resolution while decreasing the step size, which makes computation painfully long. So a feature to fix this is to have adaptive steps, which decreased the step size as the pixels got closer and closer to the middle.

Vertical line issue:

<img width="366" height="432" alt="image" src="https://github.com/user-attachments/assets/682dc687-cb44-42c2-adc5-0ae7be32f15a" />

#Gallery
These images are available in the images/ foler

4K, 4 AA
<img width="3840" height="2160" alt="4K_with_4AA" src="https://github.com/user-attachments/assets/1e460389-837c-4c12-9eec-abbcd0dee7b8" />

1080p, no AA
<img width="1920" height="1080" alt="wormhole_1080p" src="https://github.com/user-attachments/assets/d577ec76-0527-44e1-87d2-d01be0717f92" />

4K, no AA
<img width="3840" height="2160" alt="wormhole_4K" src="https://github.com/user-attachments/assets/03d9bd93-2ad6-4f34-9d6a-ddd6d847d777" />

1080p, 2 AA
<img width="1920" height="1080" alt="1080p_2aa" src="https://github.com/user-attachments/assets/ea4332d2-6b59-45e6-87f8-b03ce6a76117" />

1080p, 4 AA
<img width="1920" height="1080" alt="1080p_4aa" src="https://github.com/user-attachments/assets/5d44deff-ffab-4dc3-9b85-43e4fb753eee" />

Upon closer inspection (between 1080p 4AA and 1080p 2AA) 4AA has more stars

4AA:
<img width="477" height="659" alt="image" src="https://github.com/user-attachments/assets/41914c47-b759-4286-874a-baa58c9be7f8" />

2AA:
<img width="470" height="714" alt="image" src="https://github.com/user-attachments/assets/797d76be-8a3c-4fb4-8301-45d6e7b8813d" />

# Specs
My specs are:

RTX 5070ti

AMD Ryzen 7 5800XT

32GB RAM

2TB SSD

# Skyboxes
I used skyboxes from space sphere maps:
https://www.spacespheremaps.com/hdr-spheremaps/ 

and from NASA's SVS sky maps:
https://svs.gsfc.nasa.gov/search/?keywords=Sky%20Map
NASA/Goddard Space Flight Center Scientific Visualization Studio. Gaia DR2: ESA/Gaia/DPAC. Constellation figures based on those developed for the IAU by Alan MacRobert of Sky and Telescope magazine (Roger Sinnott and Rick Fienberg).

# Runtimes
Runtimes are for one frame, either of an image or video

200x200 on cpu: 40 minutes

200x200 on gpu: less than a second

1080p on gpu, no AA, just on wormhole_cuda.py: 1 minute

1080p on gpu, 2AA, wormhole_video_withAA.py: 2 minutes

1080p on gpu, 4AA, wormhole_video_withAA.py: 4 minutes

I'm not sure if runtimes will vary depending on hardware, but I'm assuming they will, so these runtimes are not exact. Also, they vary depending on whether or not adaptive steps are active.

# Contact
My email is v1modeler@gmail.com 

Please contact me for any questions or feedback*

I might respond a little late, as I don't really check my emails that often*
