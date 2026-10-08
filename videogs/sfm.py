"""Official pycolmap CPU SfM, distortion correction, registration quality gates."""
from pathlib import Path
import json
import numpy as np
from .video import write_json


def reconstruct(frames, work, camera_model='SIMPLE_RADIAL', camera_params='', overlap=12,
                min_registration=.75, threads=4):
    import pycolmap as pc
    frames = Path(frames); work = Path(work); work.mkdir(parents=True, exist_ok=True)
    database = work/'database.db'; sparse = work/'sparse'; sparse.mkdir(exist_ok=True)
    record = json.loads((frames/'frames.json').read_text()); names = [r['name'] for r in record['frames']]
    reader = pc.ImageReaderOptions()
    if camera_params: reader.camera_params = camera_params
    extraction = pc.FeatureExtractionOptions(); extraction.num_threads = threads
    extraction.max_image_size = 1600; extraction.sift.max_num_features = 8192
    # CPU works on macOS and Colab wheels without needing a CUDA COLMAP build.
    pc.extract_features(str(database), str(frames/'images'), image_names=names,
                        camera_mode=pc.CameraMode.SINGLE, camera_model=camera_model,
                        reader_options=reader, extraction_options=extraction, device=pc.Device.cpu)
    matching = pc.FeatureMatchingOptions(); matching.num_threads = threads
    pairing = pc.SequentialPairingOptions(); pairing.overlap = overlap; pairing.loop_detection = False
    pc.match_sequential(str(database), matching_options=matching, pairing_options=pairing, device=pc.Device.cpu)
    options = pc.IncrementalPipelineOptions(); options.num_threads = threads; options.random_seed = 7
    options.min_model_size = 10; options.mapper.init_min_tri_angle = 8.
    options.mapper.init_min_num_inliers = 80
    models = pc.incremental_mapping(str(database), str(frames/'images'), str(sparse), options=options)
    # Short-video fallback connects non-neighbouring views. Never invent camera poses.
    best = max(models.values(), key=lambda m: m.num_reg_images(), default=None)
    retries = []
    if best is None or best.num_reg_images()/len(names) < min_registration:
        if len(names) <= 240:
            retries.append('exhaustive_matching')
            pc.match_exhaustive(str(database), matching_options=matching, device=pc.Device.cpu)
            retry = work/'sparse-retry'; retry.mkdir(exist_ok=True)
            newer = pc.incremental_mapping(str(database), str(frames/'images'), str(retry), options=options)
            choices = list(models.values())+list(newer.values())
            best = max(choices, key=lambda m: m.num_reg_images(), default=None)
    if best is None:
        raise RuntimeError('COLMAP could not initialize 3D. Check camera translation, overlap, blur and scene texture; pure rotation cannot recover depth.')
    registered = {im.name for im in best.images.values() if im.has_pose}
    skipped = [n for n in names if n not in registered]
    quality = dict(pycolmap_version=pc.__version__, extracted_frames=len(names),
                   registered_frames=len(registered), registration_ratio=len(registered)/len(names),
                   sparse_points=best.num_points3D(), mean_reprojection_error=float(best.compute_mean_reprojection_error()),
                   skipped_frames=skipped, retries=retries, camera_model=camera_model,
                   pose_source='estimated_from_RGB_only', scale='arbitrary monocular SfM scale',
                   evaluation='RGB held out from Gaussian optimization; all frames participate in offline SfM')
    write_json(work/'sfm-quality.json', quality)
    if len(registered) < 12 or quality['registration_ratio'] < min_registration or quality['sparse_points'] < 100:
        raise RuntimeError(f'Insufficient SfM coverage: {len(registered)}/{len(names)} frames, {best.num_points3D()} points. See sfm-quality.json. Capture more overlapping translated views; a separate output directory preserves this diagnostic.')
    selected = work/'selected-model'; selected.mkdir(exist_ok=True); best.write(str(selected))
    scene = work/'scene'
    pc.undistort_images(str(scene), str(selected), str(frames/'images'),
                        undistort_options=pc.UndistortCameraOptions({'max_image_size': 1600}))
    surviving = [r for r in record['frames'] if r['name'] in registered]
    for i, row in enumerate(surviving):
        row['index'] = i; row['test'] = i % 8 == 7
    write_json(scene/'frames.json', dict(frames=surviving, source_video_sha256=record['video_sha256'],
               test_every=8, pose_source='offline_RGB_SfM', units='arbitrary scale'))
    return scene, quality


def usable_stages(scene, requested):
    """Do not start optimization before enough triangulated training observations."""
    n = len(scene['images'])
    for stages in range(min(requested, n//8), 0, -1):
        ends = [round(n*i/stages) for i in range(stages+1)]
        if all(sum(not im['test'] for im in scene['images'][a:b]) >= 2 for a, b in zip(ends, ends[1:])):
            if np.count_nonzero(scene['release'] < ends[1]) >= 100: return stages
    raise ValueError('Not enough triangulated anchors to initialize Gaussian mapping.')
