"""Python fallback for bbox_overlaps when compiled extension is unavailable."""
import numpy as np


def bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between each pair of boxes and query_boxes.

    This mirrors the signature of the compiled cython_bbox.bbox_overlaps used
    throughout the codebase and serves as a drop-in replacement when the
    compiled extension for the current Python version is missing.

    Args:
        boxes (np.ndarray): Array of shape (N, 4) with [x1, y1, x2, y2].
        query_boxes (np.ndarray): Array of shape (K, 4) with [x1, y1, x2, y2].

    Returns:
        np.ndarray: IoU matrix of shape (N, K).
    """
    boxes = boxes.astype(np.float32, copy=False)
    query_boxes = query_boxes.astype(np.float32, copy=False)

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        query_box = query_boxes[k]
        q_x1, q_y1, q_x2, q_y2 = query_box

        width = np.minimum(boxes[:, 2], q_x2) - np.maximum(boxes[:, 0], q_x1) + 1
        height = np.minimum(boxes[:, 3], q_y2) - np.maximum(boxes[:, 1], q_y1) + 1

        valid = (width > 0) & (height > 0)
        if not np.any(valid):
            continue

        intersection = width[valid] * height[valid]
        box_areas = (boxes[valid, 2] - boxes[valid, 0] + 1) * (boxes[valid, 3] - boxes[valid, 1] + 1)
        query_area = (q_x2 - q_x1 + 1) * (q_y2 - q_y1 + 1)
        union = box_areas + query_area - intersection

        overlaps[valid, k] = intersection / union

    return overlaps
