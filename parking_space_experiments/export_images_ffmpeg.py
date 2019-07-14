import os
import subprocess as sp
from tqdm import tqdm

videos = os.listdir('videos')

for video in tqdm(videos):
    path = os.path.join('videos', video)

    output_folder = 'images_ffmpeg/{}'.format(video[:-4])
    os.makedirs(output_folder, exist_ok=True)

    command = 'ffmpeg -i {} -vf fps=1 {}/%04d.jpg'.format(path, output_folder)

    sp.call(command, shell=True)
