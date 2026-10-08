"""Timestamp-aware decoding; no shell, sensor depth, or input camera poses."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import math
import numpy as np
from PIL import Image


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sharpness(gray):
    a = gray.astype(np.float32)
    lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4*a[1:-1, 1:-1]
    return float(lap.var())


def rotation_rgb(rgb, degrees):
    # Display-matrix angles follow counter-clockwise convention in PyAV.
    turns = round(degrees / 90)
    if abs(degrees - turns*90) > 1:
        raise ValueError('Unsupported non-right-angle video rotation; transcode first.')
    return np.ascontiguousarray(np.rot90(rgb, turns % 4))


def extract(video, output, fps=3., max_frames=180, width=960, start=0., duration=None,
            blur_threshold=0., duplicate_threshold=1., rotate=None):
    import av
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f'Video file not found: {video}')
    if fps <= 0 or max_frames < 12 or width < 64 or start < 0 or (duration is not None and duration <= 0):
        raise ValueError('Require fps>0, max_frames>=12, width>=64, start>=0, duration>0.')
    out = Path(output); images = out/'images'; images.mkdir(parents=True, exist_ok=True)
    rows = []; rejected = {'blur': 0, 'duplicate': 0}; last_gray = None; next_time = start
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError('The input contains no video stream.')
        stream = container.streams.video[0]; stream.thread_type = 'AUTO'
        rate = float(stream.average_rate or 30)
        length = float(stream.duration*stream.time_base) if stream.duration is not None else (
            container.duration / av.time_base if container.duration else None)
        span = duration if duration is not None else (max(0., length-start) if length else None)
        if length is not None and span is not None: span = min(span, max(0., length-start))
        interval = max(1/fps, span/max(1, max_frames-1)) if span else 1/fps
        origin = None
        for source_index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else source_index/rate
            if origin is None: origin = timestamp
            timestamp -= origin
            if span is not None and timestamp > start+span+1e-6: break
            if timestamp+1e-6 < next_time: continue
            next_time = timestamp+interval
            rgb = frame.to_ndarray(format='rgb24')
            angle = rotate if rotate is not None else float(getattr(frame, 'rotation', 0) or stream.metadata.get('rotate', 0))
            rgb = rotation_rgb(rgb, angle)
            pic = Image.fromarray(rgb); w = min(width, pic.width); h = max(2, round(pic.height*w/pic.width))
            pic = pic.resize((w, h), Image.Resampling.LANCZOS)
            gray = np.asarray(pic.convert('L').resize((160, 120)))
            score = sharpness(gray)
            if score < blur_threshold:
                rejected['blur'] += 1; continue
            difference = float(np.abs(gray.astype(float)-last_gray).mean()) if last_gray is not None else None
            if difference is not None and difference < duplicate_threshold and timestamp-rows[-1]['timestamp'] < 2:
                rejected['duplicate'] += 1; continue
            index = len(rows); name = f'{index:06d}.jpg'; pic.save(images/name, quality=95)
            rows.append(dict(index=index, name=name, source_frame=source_index, timestamp=timestamp,
                             width=w, height=h, sharpness=score, difference=difference, rotation=angle))
            last_gray = gray.astype(float)
            if len(rows) >= max_frames: break
    if len(rows) < 12:
        raise ValueError(f'Only {len(rows)} usable frames. Use a longer video, higher --fps, or lower blur filtering.')
    record = dict(video_name=video.name, video_sha256=sha256(video), requested_fps=fps,
                  interval_seconds=interval, decoded_dimensions=[rows[0]['width'], rows[0]['height']],
                  rejected=rejected, frames=rows, has_depth=False, input_camera_poses=False)
    write_json(out/'frames.json', record)
    return record


def encode_preview(rows, images, path, fps=None):
    """H.264 browser preview; preserve selected frame timestamps when fps is None."""
    import av
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=Fraction(fps or 30))
        with Image.open(Path(images)/rows[0]['name']) as first:
            stream.width = first.width//2*2; stream.height = first.height//2*2
        stream.pix_fmt = 'yuv420p'; stream.time_base = Fraction(1, 1000)
        stream.options = {'crf': '21', 'preset': 'fast'}
        base = rows[0]['timestamp']
        for i, row in enumerate(rows):
            with Image.open(Path(images)/row['name']) as pic:
                frame = av.VideoFrame.from_image(pic.convert('RGB').resize((stream.width, stream.height)))
            frame.pts = round((i/fps if fps else row['timestamp']-base)*1000)
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
