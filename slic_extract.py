import os
from skimage import io
from skimage.segmentation import slic
from skimage.util import img_as_ubyte
from skimage.color import label2rgb
import matplotlib.pyplot as plt

def extract_and_save_superpixels(input_folder, output_folder, n_segments=250, compactness=10, sigma=1):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    for filename in os.listdir(input_folder):

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            file_path = os.path.join(input_folder, filename)

            img = io.imread(file_path)
            

            segments = slic(img, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1)

            segmented_image = label2rgb(segments, img, kind='avg')
            

            output_file_path = os.path.join(output_folder, f"superpixel_{filename}")

            io.imsave(output_file_path, img_as_ubyte(segmented_image))
            print(f"Processed and saved: {output_file_path}")


input_folder = 'path_to_your_input_folder'  
output_folder = 'superpixels_output'  
extract_and_save_superpixels(input_folder, output_folder)
