import scannerpy as sp
import scannertools.object_detection
import scannertools.vis
import pickle

if __name__ == "__main__":
    path = '/app/videos/20190630_194819_no_sound.mp4'

    cl = sp.Client()
    video = sp.NamedVideoStream(cl, 'example', path=path)
    frames = cl.io.Input([video])
    subset = cl.streams.Range(frames, [(0, 10)])
    objects = cl.ops.DetectObjects(frame=subset, device=sp.DeviceType.GPU)
    output_stream = sp.NamedStream(cl, 'output_objects')
    output = cl.io.Output(objects, [output_stream])
    cl.run(output, sp.PerfParams.manual(
        work_packet_size = 10, io_packet_size = 100
    ), cache_mode=sp.CacheMode.Overwrite)

    for i, out in enumerate(output_stream.load()):
        if i == 0:
            objects_frame0 = out

    with open('objects_frame0.pkl', 'wb') as f:
        pickle.dump(objects_frame0, f)