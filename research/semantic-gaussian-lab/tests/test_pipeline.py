import json
import gzip
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from sglab.data import Frame, ManifestSequence, backproject
from sglab.mapping import GaussianMap, MapConfig
from sglab.synthetic import SyntheticSequence, LABELS
from sglab.tracking import InstanceTracker
from sglab.segmentation import decode_panoptic
from sglab.tum import import_tum, quaternion_matrix, nearest
from sglab.evaluation import probe_old_view


def make_frame(label=3,local=1,color=100):
    shape=(8,8); K=np.array([[30,0,3.5],[0,30,3.5],[0,0,1]],np.float32)
    return Frame(np.full((*shape,3),color,np.uint8),np.ones(shape,np.float32),K,np.eye(4,dtype=np.float32),
        np.full(shape,label,np.int32),np.full(shape,local,np.int32),np.ones(shape,np.float32),0)


class PipelineTests(unittest.TestCase):
    def test_coordinate_convention_and_invalid_depth(self):
        f=make_frame(); f.depth[:]=0; f.depth[0,0]=np.nan; f.depth[4,4]=2; f.c2w[0,3]=1
        xyz,*_=backproject(f,stride=1)
        np.testing.assert_allclose(xyz,[[1+1/30,1/30,2]],rtol=1e-6)

    def test_hard_cap_and_capacity(self):
        m=GaussianMap(LABELS,MapConfig(max_gaussians=5,stride=1,adaptive_budget=False,voxel_size=.01))
        s,_,_=m.integrate(make_frame())
        self.assertEqual(m.count,5); self.assertGreater(s["rejected"],0)
        self.assertEqual(m.data.nbytes,5*m.dtype.itemsize)

    def test_one_hit_per_voxel_per_frame(self):
        m=GaussianMap(LABELS,MapConfig(stride=1,voxel_size=1,adaptive_budget=False))
        m.integrate(make_frame()); self.assertEqual(m.data[:m.count]["hits"].max(),1)

    def test_stable_region_does_not_forget(self):
        m=GaussianMap(LABELS,MapConfig(stride=1,stable_hits=2,adaptive_budget=False))
        m.integrate(make_frame()); m.integrate(make_frame())
        before=m.data[:m.count].copy(); m.integrate(make_frame(label=4,local=0,color=250))
        stable=before["hits"]>=2
        np.testing.assert_array_equal(before["mean"][stable],m.data[:len(before)]["mean"][stable])
        np.testing.assert_array_equal(before["color"][stable],m.data[:len(before)]["color"][stable])
        np.testing.assert_array_equal(before["evidence"][stable],m.data[:len(before)]["evidence"][stable])

    def test_confidence_gates_semantics(self):
        f=make_frame(); f.confidence[:]=.1
        m=GaussianMap(LABELS,MapConfig(stride=1,adaptive_budget=False)); m.integrate(f)
        self.assertTrue(all(s==0 for s in m.packet()["semantic"]))
        self.assertEqual(len(m.tracker.tracks),0)

    def test_dynamic_person_class_excluded(self):
        m=GaussianMap(LABELS,MapConfig(stride=1,dynamic_labels=(3,),adaptive_budget=False)); m.integrate(make_frame())
        self.assertEqual(m.count,0)

    def test_tracker_preserves_identity_after_local_ids_change(self):
        t=InstanceTracker(); xyz=np.array([[0,0,1],[.15,0,1],[.3,0,1],[1,0,1],[1.15,0,1],[1.3,0,1]])
        sem=np.full(6,4); conf=np.ones(6)
        a=t.associate(xyz,sem,np.array([1,1,1,2,2,2]),0,conf)
        b=t.associate(xyz,sem,np.array([99,99,99,4,4,4]),1,conf)
        np.testing.assert_array_equal(a,b); self.assertNotEqual(a[0],a[-1])
        c=t.associate(xyz[:3],sem[:3],np.full(3,8),40,conf[:3])
        np.testing.assert_array_equal(c,a[:3])

    def test_panoptic_unknown_and_stuff(self):
        seg=np.array([[-1,3],[4,4]]); info=[{"id":3,"label_id":56,"score":.9},{"id":4,"label_id":90,"score":.8}]
        s,i,c=decode_panoptic(seg,info,seg.shape,set(range(80)))
        self.assertEqual(s[0,0],0); self.assertEqual(i[0,1],4); self.assertEqual(i[1,1],0)
        self.assertEqual(s[1,1],91); self.assertEqual(c[0,0],0)

    def test_pruning_compacts_lookup(self):
        m=GaussianMap(LABELS,MapConfig(stride=1,prune_after=1,adaptive_budget=False))
        m.integrate(make_frame()); m.frame_id=10
        pruned=m._prune(); self.assertGreater(pruned,0); self.assertEqual(m.lookup,{})
        m.integrate(make_frame()); self.assertEqual(len(m.lookup),m.count)

    def test_pose_validation(self):
        f=make_frame(); f.c2w[0,0]=-1
        with self.assertRaisesRegex(ValueError,"right-handed"): f.validate(len(LABELS))

    def test_export_and_probe(self):
        m=GaussianMap(LABELS,MapConfig(stride=1,adaptive_budget=False)); f=make_frame(); m.integrate(f)
        probe=probe_old_view(m,f); self.assertGreater(probe["coverage"],0)
        with tempfile.TemporaryDirectory() as directory:
            m.export(directory); data=np.load(Path(directory)/"map.npz",allow_pickle=False)
            self.assertEqual(len(data["gaussians"]),m.count)
            self.assertIn("property int instance_id",(Path(directory)/"map.ply").read_text())

    def test_synthetic_closed_loop_and_loader(self):
        seq=SyntheticSequence(count=3,width=32,height=24); a,b=seq.get(0),seq.get(2)
        np.testing.assert_allclose(a.c2w,b.c2w,atol=1e-6); a.validate(len(LABELS))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); Image.fromarray(a.rgb).save(root/"rgb.png"); np.save(root/"depth.npy",a.depth)
            manifest={"pose_convention":"opencv_c2w_meters","labels":LABELS,"K":a.K.tolist(),
                      "frames":[{"rgb":"rgb.png","depth":"depth.npy","c2w":a.c2w.tolist()}]}
            (root/"manifest.json").write_text(json.dumps(manifest)); f=ManifestSequence(root/"manifest.json").get(0)
            self.assertEqual(f.segmentation_source,"none"); self.assertFalse(f.semantic.any())

    def test_actual_map_replay_does_not_refuse_or_change_budget(self):
        from sglab.replay import ReplaySession
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); f=make_frame(); Image.fromarray(f.rgb).save(root/'rgb.png')
            np.save(root/'depth.npy',f.depth)
            manifest={'pose_convention':'opencv_c2w_meters','labels':LABELS,'K':f.K.tolist(),
                'frames':[{'rgb':'rgb.png','depth':'depth.npy','c2w':f.c2w.tolist()}]}
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            m=GaussianMap(LABELS,MapConfig(max_gaussians=100,stride=1,adaptive_budget=False))
            stats,_,_=m.integrate(f);stats['mapping_total_ms']=stats['update_ms']
            result=root/'result';m.export(result)
            (result/'summary.json').write_text(json.dumps({'train_frames':1,'gaussians':m.count}))
            (result/'replay').mkdir()
            snapshot={'source_frame':0,'next_frame':1,'stats':stats,'map':m.packet(),
                'trajectory':m.trajectory,'camera':f.c2w.tolist(),'objects':m.tracker.summary()}
            with gzip.open(result/'replay'/'0000.json.gz','wt') as stream:json.dump(snapshot,stream)
            session=ReplaySession(path,result)
            self.assertEqual(session.state()['map']['ids'],[])
            step=session.step();self.assertTrue(step['done'])
            self.assertEqual(step['map'],snapshot['map']);self.assertEqual(len(step['history']),1)
            self.assertIn('data:image/png;base64,',step['images']['rgb'])
            self.assertTrue(session.step()['done'])
            state=session.reset({'max_gaussians':50000})
            self.assertEqual(state['next_frame'],0);self.assertEqual(state['config']['max_gaussians'],100)

    def test_tum_import_depth_pose_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); src=root/"raw"; src.mkdir()
            Image.fromarray(np.full((480,640,3),80,np.uint8)).save(src/"rgb.png")
            Image.fromarray(np.full((480,640),5000,np.uint16)).save(src/"depth.png")
            (src/"rgb.txt").write_text("1.0 rgb.png\n"); (src/"depth.txt").write_text("1.005 depth.png\n")
            (src/"groundtruth.txt").write_text("1.0 1 2 3 0 0 0 1\n")
            path=import_tum(src,root/"converted",1,1)
            f=ManifestSequence(path).get(0); self.assertEqual(f.rgb.shape,(240,320,3))
            np.testing.assert_array_equal(f.depth,1); self.assertEqual(f.K[0,0],262.5)
            np.testing.assert_allclose(f.c2w[:3,3],0); self.assertEqual(f.c2w[1,1],-1)
        np.testing.assert_array_equal(quaternion_matrix([0,0,0,1]),np.eye(3))
        self.assertIsNone(nearest([(1,[])],3))


if __name__=="__main__": unittest.main()
