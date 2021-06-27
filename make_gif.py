import os
import imageio

png_dir = '/home/jihoon/Repositories/InfoGAN/results/infogan_True_noise_True_code_dl_True/generated'
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('figure.gif', images)