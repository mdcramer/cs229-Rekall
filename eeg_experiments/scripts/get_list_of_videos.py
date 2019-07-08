# Script to get a list of all the videos and save them in 'wmvs.txt'

import os

root_folder = '/mnt/data2/danfu/rekall_experiments/eeg_experiments'
video_folder = 'eeg_video'

wmv_files = [
    os.path.join(tup[0], f)
    for tup in os.walk(os.path.join(root_folder, video_folder))
    for f in tup[2]
    if f[-3:] == 'WMV'
]

with open(os.path.join(root_folder, 'wmvs.txt'), 'w') as f:
    for filename in wmv_files:
        f.write('{}\n'.format(filename)) 
