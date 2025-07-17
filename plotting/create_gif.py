import imageio.v2 as imageio
import os
import glob

def create_gif_from_folder(image_folder, output_folder="gifs"):

    image_files = sorted(glob.glob(os.path.join(image_folder, 't*_contour.png')))
    output_gif = os.path.join(output_folder, f'{os.path.basename(image_folder)}.gif')

    with imageio.get_writer(output_gif, mode='I', duration=0.1) as writer:  # duration in Sekunden pro Frame
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"GIF gespeichert unter: {output_gif}")
