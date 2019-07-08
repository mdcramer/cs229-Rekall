# Convert WMV files to mp4 files using ffmpeg. Expects list of wmvs in wmvs.txt.

import subprocess as sp
import os
from tqdm import tqdm

root_folder = '/mnt/data2/danfu/rekall_experiments/eeg_experiments'

wmv_files = []
with open(os.path.join(root_folder, 'wmvs.txt'), 'r') as f:
    for filename in f.readlines():
        wmv_files.append(filename.strip())

mp4_files = [
    f[:-3] + 'mp4'
    for f in wmv_files
]

for wmv, mp4 in tqdm(zip(wmv_files, mp4_files), total=len(wmv_files)):
    sp.call(['ffmpeg', '-i', wmv, mp4])
