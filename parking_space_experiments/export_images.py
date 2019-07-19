import hwang, storehouse
import os
from tqdm import tqdm
import subprocess
import shlex
import json
from PIL import Image

videos = os.listdir('/app/videos')

for video in tqdm(videos):
    path = os.path.join('/app/videos', video)
    
    cmd = "ffprobe -v quiet -print_format json -show_streams %s" % path;
    outp = subprocess.check_output(shlex.split(cmd)).decode("utf-8")
    streams = json.loads(outp)["streams"]
    video_stream = [s for s in streams if s["codec_type"] == "video"][0]

    [num, denom] = map(int, video_stream["r_frame_rate"].split('/'))
    fps = float(num) / float(denom)
    num_frames = video_stream["nb_frames"]
    width = video_stream["width"]
    height = video_stream["height"]
    
    frame_nums = list(range(0, int(num_frames), 30))
    
    backend = storehouse.StorageBackend.make_from_config(
        storehouse.StorageConfig.make_posix_config()
    )
    dec = hwang.Decoder(storehouse.RandomReadFile(backend, path))
    
    frames = dec.retrieve(frame_nums)
    
    os.makedirs('/app/images/{}'.format(video[:-4]), exist_ok=True)
    
    for i, frame in enumerate(frames):
        im = Image.fromarray(frame)
        im.save('/app/images/{}/{:04d}.jpg'.format(video[:-4], i))