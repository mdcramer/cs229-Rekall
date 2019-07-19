import scannerpy as sp
import scannertools_caffe
import scannertools.vis
import pickle

if __name__ == "__main__":
    path = '/app/eeg_video/sample_data/20698/FA444085.VOR/44408500.mp4'

    cl = sp.Client()
    video = sp.NamedVideoStream(cl, 'example', path=path)
    frames = cl.io.Input([video])
    subset = cl.streams.Range(frames, [(0, 10)])
    pose = cl.ops.OpenPose(
        frame=frames,
        device=sp.DeviceType.GPU,
        pose_num_scales=6,
        pose_scale_gap=0.16,
        compute_hands=True,
        hand_num_scales=6,
        hand_scale_gap=0.16,
        compute_face=True,
        batch=5
    )
    output_stream = sp.NamedStream(cl, 'output_poses')
    output = cl.io.Output(pose, [output_stream])
    cl.run(output, sp.PerfParams.manual(
        work_packet_size = 10, io_packet_size = 100
    ), cache_mode=sp.CacheMode.Overwrite)

    for i, out in enumerate(output_stream.load()):
        if i == 0:
            poses_frame0 = out

    with open('poses_frame0.pkl', 'wb') as f:
        pickle.dump(poses_frame0, f)

