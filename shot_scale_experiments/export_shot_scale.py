import pickle
import os
import scannertools as st
from tqdm import tqdm
from query.models import Video
from PIL import Image

st.init_storage(os.environ['BUCKET'])

OUTPUT_FOLDER = '/app/data/shot_scale/images/'

with open('/app/data/shot_scales_and_conv_idioms_samples/' + 
          'shot_scale_labels_and_rekall_accuracy_val_test.pkl', 'rb') as f:
    data = pickle.load(f)
    
val_and_test = sorted(data['val_set'] + data['test_set'])

for (video_id, start, end) in tqdm(val_and_test):
    dst_folder = os.path.join(OUTPUT_FOLDER, str(video_id))
    os.makedirs(dst_folder, exist_ok=True)
    
    frame_nums = list(range(start, end + 1))
    
    frames = Video.objects.get(id=video_id).for_scannertools().frames(frame_nums)
    
    for frame, frame_num in zip(frames, frame_nums):
        im = Image.fromarray(frame)
        im.save(os.path.join(dst_folder, '{}.jpg'.format(frame_num)))