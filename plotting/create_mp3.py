import cv2
import os
import glob

# TODO
image_folder = r"C:/Users/rabea/Documents/Numerik/plots/plots_CFL0p30_Re400"

image_files = sorted(glob.glob(os.path.join(image_folder, 't*_contour.png')))

if not image_files:
    raise FileNotFoundError("Keine Bilder gefunden!")

first_frame = cv2.imread(image_files[0])
height, width, _ = first_frame.shape
frame_size = (width, height)

output_path = os.path.join(image_folder, 'simulation_video.mp4')
fps = 10 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

for filename in image_files:
    frame = cv2.imread(filename)
    video.write(frame)

video.release()
print(f"Video gespeichert unter: {output_path}")
