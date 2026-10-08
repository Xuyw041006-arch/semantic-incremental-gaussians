"""Explicit OpenCV camera convention: x right, y down, z forward; metres."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from PIL import Image


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    c2w: np.ndarray
    semantic: np.ndarray
    instance: np.ndarray  # frame-local positive IDs, zero for stuff/unknown
    confidence: np.ndarray
    timestamp: float
    segmentation_source: str = "provided"

    def validate(self, classes):
        h, w = self.depth.shape
        if self.rgb.shape != (h, w, 3) or self.rgb.dtype != np.uint8:
            raise ValueError("rgb must be HxWx3 uint8, aligned to depth")
        for name in ("semantic", "instance", "confidence"):
            if getattr(self, name).shape != (h, w):
                raise ValueError(f"{name} must be depth-aligned HxW")
        if self.semantic.dtype.kind not in "iu" or self.instance.dtype.kind not in "iu":
            raise ValueError("semantic and instance must be integer IDs")
        if np.any(self.semantic < 0) or np.any(self.semantic >= classes) or np.any(self.instance < 0):
            raise ValueError("invalid semantic / instance ID")
        if not np.isfinite(self.confidence).all() or np.any(self.confidence < 0) or np.any(self.confidence > 1):
            raise ValueError("confidence must be finite in [0,1]")
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all() or self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("invalid camera intrinsics")
        if not np.allclose(self.K[2], [0, 0, 1]) or abs(self.K[0, 1]) > 1e-6:
            raise ValueError("only zero-skew pinhole intrinsics are supported")
        if self.c2w.shape != (4, 4) or not np.isfinite(self.c2w).all() or not np.allclose(self.c2w[3], [0, 0, 0, 1]):
            raise ValueError("c2w must be a finite homogeneous 4x4 matrix")
        r = self.c2w[:3, :3]
        if not np.allclose(r.T @ r, np.eye(3), atol=2e-3) or not np.isclose(np.linalg.det(r), 1, atol=2e-3):
            raise ValueError("c2w rotation must be right-handed and orthonormal")


class ManifestSequence:
    """Load a pre-aligned RGB-D sequence; no hidden depth or pose estimation."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("pose_convention") != "opencv_c2w_meters":
            raise ValueError("manifest pose_convention must be opencv_c2w_meters")
        self.labels = self.manifest["labels"]
        if not self.labels or self.labels[0]["name"] != "unknown":
            raise ValueError("labels[0] must be unknown")
        self.frames = self.manifest["frames"]
        if not self.frames:
            raise ValueError("empty sequence")
        self.name = self.manifest.get("name", self.path.stem)
        self.description = self.manifest.get("description", "真实 RGB-D / 输入位姿")

    def __len__(self):
        return len(self.frames)

    def _load(self, path):
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("dataset paths must stay inside the manifest directory")
        if p.suffix == ".npy":
            return np.load(p, allow_pickle=False)
        return np.asarray(Image.open(p))

    def get(self, index):
        f = self.frames[index]
        rgb = self._load(f["rgb"])[..., :3].astype(np.uint8)
        depth = self._load(f["depth"]).astype(np.float32) * float(self.manifest.get("depth_scale", 1))
        shape = depth.shape
        semantic = self._load(f["semantic"]).astype(np.int32) if "semantic" in f else np.zeros(shape, np.int32)
        instance = self._load(f["instance"]).astype(np.int32) if "instance" in f else np.zeros(shape, np.int32)
        confidence = self._load(f["confidence"]).astype(np.float32) if "confidence" in f else (semantic > 0).astype(np.float32)
        frame = Frame(rgb, depth, np.array(f.get("K", self.manifest.get("K")), np.float32),
                      np.array(f["c2w"], np.float32), semantic, instance, confidence,
                      float(f.get("timestamp", index / 10)), f.get("segmentation_source", "provided" if "semantic" in f else "none"))
        frame.validate(len(self.labels))
        return frame


def backproject(frame, stride=3, near=0.15, far=8.0):
    v, u = np.mgrid[0:frame.depth.shape[0]:stride, 0:frame.depth.shape[1]:stride]
    z = frame.depth[v, u]
    valid = np.isfinite(z) & (z > near) & (z < far)
    u, v, z = u[valid], v[valid], z[valid]
    xyz = np.stack(((u-frame.K[0, 2])*z/frame.K[0, 0],
                    (v-frame.K[1, 2])*z/frame.K[1, 1], z), axis=1)
    xyz = xyz @ frame.c2w[:3, :3].T + frame.c2w[:3, 3]
    return xyz.astype(np.float32), frame.rgb[v, u].astype(np.float32)/255, frame.semantic[v, u], frame.instance[v, u], frame.confidence[v, u], z
