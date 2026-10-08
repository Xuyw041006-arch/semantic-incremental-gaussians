"""Task priorities and semantic boundaries. CLIP similarity is not importance."""
from collections import Counter
import re
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

DEFAULT_IMPORTANT = 'monitor,keyboard,mouse,cup,bottle,book,laptop'
BACKGROUND = {'wall', 'floor', 'ceiling', 'background', 'window', 'door'}


def parse_queries(text):
    return [s.strip().lower() for s in re.split('[,;，；]', text or '') if s.strip()]


def boundary_band(labels, radius=2):
    """Include both sides of actual region transitions, excluding unknown/unknown."""
    labels = np.asarray(labels)
    return ((maximum_filter(labels, size=3) != minimum_filter(labels, size=3)) &
            (maximum_filter(labels, size=3) > 0)) if radius <= 0 else maximum_filter(
                (maximum_filter(labels, size=3) != minimum_filter(labels, size=3)) &
                (maximum_filter(labels, size=3) > 0), size=2*radius+1).astype(bool)


def node_priorities(nodes, important=DEFAULT_IMPORTANT, query_features=None, vocabulary_features=None):
    queries = parse_queries(important)
    result = {}
    for key, node in nodes.items():
        votes = Counter(s.lower() for s in node.get('names', []))
        name = votes.most_common(1)[0][0] if votes else 'unknown'
        exact = any(name == q or name.startswith(q+' ') for q in queries)
        similarity = None;label_similarity=None
        if query_features is not None and node.get('descriptors'):
            f = np.asarray(node['descriptors'], np.float32)
            f /= np.linalg.norm(f, axis=1, keepdims=True).clip(1e-8)
            similarity = float((f @ query_features.T).max())
            if vocabulary_features is not None and name in vocabulary_features:
                label_similarity=float((f @ vocabulary_features[name]).max())
        reliable = node.get('confidence', 0) >= .75
        # Background classification takes precedence over a weak open-vocabulary match.
        background = reliable and name in BACKGROUND and votes[name]/max(1,sum(votes.values())) >= .65
        matched = exact or (not background and similarity is not None and similarity >= .28 and
                           label_similarity is not None and similarity >= label_similarity+.02)
        result[str(key)] = dict(name=name, priority=3. if reliable and matched else (0. if background else 1.),
                                background=background, clip_query_cosine=similarity)
    # Candidate fine regions inherit their object's task importance.
    for key, node in nodes.items():
        parent = result.get(str(node.get('parent', 0)))
        if parent and parent['priority'] > result[key]['priority']:
            result[key]['priority'] = parent['priority']; result[key]['background'] = False
    return result


def anchor_priority(ids, confidence, priorities):
    ids = np.asarray(ids); confidence = np.asarray(confidence)
    limit = max([int(k) for k in priorities] + [int(ids.max(initial=0))]) + 1
    lut = np.ones(limit, np.float32); bg = np.zeros(limit, bool)
    for key, value in priorities.items():
        lut[int(key)] = value['priority']; bg[int(key)] = value['background']
    good = (ids > 0) & (confidence >= .75)
    p = np.where(good, lut[ids], 1.).max(axis=0)
    # Unknowns are never silently relabelled as background.
    background = good[0] & bg[ids[0]] & ~((good & (lut[ids] >= 2)).any(axis=0))
    p[background] = 0.
    return p.astype(np.float32), background


def descriptor_table(nodes, dimensions=16):
    """Fixed projection: cannot leak future-view PCA statistics into earlier stages."""
    n = max([int(k) for k in nodes] + [0]) + 1
    table = np.zeros((n, dimensions), np.float32)
    projection = np.random.default_rng(321).normal(size=(512, dimensions)).astype(np.float32)
    projection /= np.sqrt(dimensions)
    for key, node in nodes.items():
        if node.get('descriptors'):
            f = np.mean(node['descriptors'], axis=0).astype(np.float32) @ projection
            table[int(key)] = f / max(float(np.linalg.norm(f)), 1e-8)
    return table
