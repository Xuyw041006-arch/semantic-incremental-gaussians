import io,json,struct,tarfile,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from videogs.video import extract,encode_preview,rotation_rgb
from videogs.sfm import usable_stages
from videogs.deliver import export_ply
from videogs.demo import make_demo
from hglab.colmap import read_scene


class VideoPipelineTest(unittest.TestCase):
    def test_decode_bounded_full_span_with_real_pts_and_space_in_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[];images=root/'images';images.mkdir()
            rng=np.random.default_rng(4)
            for i in range(60):
                name=f'{i:06d}.png';Image.fromarray(rng.integers(0,255,(64,96,3),dtype=np.uint8)).save(images/name)
                rows.append(dict(name=name,timestamp=i*.1))
            movie=root/'video with space.mp4';encode_preview(rows,images,movie)
            record=extract(movie,root/'extract',fps=10,max_frames=15,width=96,duplicate_threshold=0)
            times=[r['timestamp'] for r in record['frames']]
            self.assertGreaterEqual(len(times),12);self.assertLessEqual(len(times),15)
            self.assertGreater(times[-1],5.);self.assertTrue(np.all(np.diff(times)>0))
            self.assertFalse(record['input_camera_poses']);self.assertFalse(record['has_depth'])

    def test_short_static_clip_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(96,64)).save(root/'frame.png')
            encode_preview([dict(name='frame.png',timestamp=i*.1) for i in range(20)],root,root/'static.mp4')
            with self.assertRaisesRegex(ValueError,'usable frames'):
                extract(root/'static.mp4',root/'out')

    def test_rotation_preserves_pixels(self):
        rgb=np.arange(18).reshape(2,3,3)
        np.testing.assert_array_equal(rotation_rgb(rotation_rgb(rgb,90),-90),rgb)

    def test_http_video_ranges_enable_seeking_and_reject_past_eof(self):
        import functools,http.client,http.server,threading
        from videogs.server import PlaybackHandler
        with tempfile.TemporaryDirectory() as tmp:
            raw=bytes(range(100));(Path(tmp)/'clip.mp4').write_bytes(raw)
            class QuietHandler(PlaybackHandler):
                def log_message(self,*args):pass
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=tmp))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                c=http.client.HTTPConnection('127.0.0.1',server.server_port)
                for header,start,end in [('bytes=20-39',20,39),('bytes=90-',90,99),('bytes=-7',93,99)]:
                    c.request('GET','/clip.mp4',headers={'Range':header});r=c.getresponse()
                    self.assertEqual(r.status,206);self.assertEqual(r.getheader('Accept-Ranges'),'bytes')
                    self.assertEqual(r.getheader('Content-Range'),f'bytes {start}-{end}/100')
                    self.assertEqual(r.read(),raw[start:end+1])
                c.request('GET','/clip.mp4',headers={'Range':'bytes=100-'});r=c.getresponse()
                self.assertEqual(r.status,416);self.assertEqual(r.getheader('Content-Range'),'bytes */100');r.read()
                c.request('HEAD','/clip.mp4',headers={'Range':'bytes=0-1'});r=c.getresponse()
                self.assertEqual(r.status,206);self.assertEqual(r.getheader('Content-Length'),'2');self.assertEqual(r.read(),b'')
                c.close()
            finally:server.shutdown();server.server_close();thread.join()

    def test_unsorted_recording_archive_uses_only_rgb_and_orders_video_pts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'tum-rgb-source';cache.mkdir()
            with tarfile.open(cache/'desk.tgz','w:gz') as tar:
                for i in np.random.default_rng(4).permutation(40):
                    b=io.BytesIO();Image.new('RGB',(96,64),(int(i)*5,90,130)).save(b,format='PNG')
                    raw=b.getvalue();entry=tarfile.TarInfo(f'dataset/rgb/{100+i*.1:.6f}.png');entry.size=len(raw)
                    tar.addfile(entry,io.BytesIO(raw))
                raw=b'not a usable camera pose';entry=tarfile.TarInfo('dataset/groundtruth.txt');entry.size=len(raw)
                tar.addfile(entry,io.BytesIO(raw))
            output=root/'demo.mp4';make_demo(output,seconds=2.)
            import av
            with av.open(str(output)) as c:times=[float(f.time) for f in c.decode(video=0)]
            self.assertEqual(len(times),21);self.assertTrue(np.all(np.diff(times)>0))
            self.assertFalse((cache/'groundtruth.txt').exists())

    def test_stage_warmup_reduces_chunks(self):
        scene=dict(images=[dict(test=i%8==7) for i in range(32)],release=np.full(120,10))
        self.assertEqual(usable_stages(scene,6),3)
        with self.assertRaisesRegex(ValueError,'anchors'):
            usable_stages(dict(images=scene['images'],release=np.full(120,99)),6)

    def test_ply_uses_log_scale_logit_opacity_and_channel_major_sh(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'g.ply';values=dict(means=np.array([[1.,2.,3.]]),scales=np.array([[.1,.2,.3]]),
                quats=np.array([[2.,0,0,0]]),opacity=np.array([.8]),sh=np.arange(12,dtype=float).reshape(1,4,3))
            export_ply(p,values,np.array([[13],[17]]));header,raw=p.read_bytes().split(b'end_header\n')
            fields=[line.split()[-1].decode() for line in header.splitlines() if line.startswith(b'property float')]
            numbers=np.frombuffer(raw[:len(fields)*4],dtype='<f4');lookup=dict(zip(fields,numbers))
            self.assertAlmostEqual(lookup['scale_0'],np.log(.1),places=6)
            self.assertAlmostEqual(lookup['opacity'],np.log(4),places=6)
            self.assertEqual(lookup['rot_0'],1)
            self.assertEqual([lookup[f'f_rest_{i}'] for i in range(9)],[3,6,9,4,7,10,5,8,11])
            self.assertEqual(struct.unpack('<ii',raw[-8:]),(13,17))

    def test_colmap_chronology_split_and_two_training_observation_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sparse=root/'sparse';sparse.mkdir();images=root/'images';images.mkdir()
            with (sparse/'cameras.bin').open('wb') as f:
                f.write(struct.pack('<QiiQQdddd',1,1,1,96,64,80.,80.,48.,32.))
            frames=[dict(name=f'{i:06d}.jpg',timestamp=i*.4,test=i==1) for i in range(4)]
            (root/'frames.json').write_text(json.dumps(dict(frames=frames)))
            with (sparse/'images.bin').open('wb') as f:
                f.write(struct.pack('<Q',4))
                for i in range(4):
                    Image.new('RGB',(96,64)).save(images/frames[i]['name'])
                    f.write(struct.pack('<idddddddi',i+1,1.,0.,0.,0.,-i*.1,0.,0.,1))
                    f.write(frames[i]['name'].encode()+b'\0');f.write(struct.pack('<Q',4))
                    for j in range(4):f.write(struct.pack('<ddq',48.,32.,j+1))
            with (sparse/'points3D.bin').open('wb') as f:
                f.write(struct.pack('<Q',4))
                for j in range(4):
                    f.write(struct.pack('<QdddBBBdQ',j+1,j*.01,0.,2.,120,130,140,0.,4))
                    for i in range(4):f.write(struct.pack('<ii',i+1,j))
            scene=read_scene(root,96)
            np.testing.assert_array_equal(scene['release'],np.full(4,2))
            self.assertEqual([im['test'] for im in scene['images']],[False,True,False,False])
            self.assertEqual(scene['images'][2]['timestamp'],.8)

if __name__=='__main__':unittest.main()
