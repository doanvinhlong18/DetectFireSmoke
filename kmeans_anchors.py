# kmeans_anchors.py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import argparse


def iou_numpy(box_wh, clusters_wh):
    """
    Calculates Intersection over Union (IoU) between a box and multiple cluster centroids.
    Args:
        box_wh (np.ndarray): Numpy array of shape (2,) representing [width, height].
        clusters_wh (np.ndarray): Numpy array of shape (k, 2) representing k centroids.
    Returns:
        np.ndarray: Numpy array of shape (k,) containing IoU values.
    """
    # Calculate intersection areas
    inter_w = np.minimum(box_wh[0], clusters_wh[:, 0])
    inter_h = np.minimum(box_wh[1], clusters_wh[:, 1])
    inter_area = inter_w * inter_h

    # Calculate union areas
    box_area = box_wh[0] * box_wh[1]
    clusters_area = clusters_wh[:, 0] * clusters_wh[:, 1]
    union_area = box_area + clusters_area - inter_area

    # Avoid division by zero
    return inter_area / (union_area + 1e-9)


def kmeans(boxes_wh, k, max_iters=1000):
    """
    Performs K-Means clustering on bounding box dimensions using IoU as the distance metric.
    Args:
        boxes_wh (np.ndarray): Numpy array of shape (n, 2) for n boxes.
        k (int): The number of clusters (anchors) to generate.
        max_iters (int): Maximum number of iterations.
    Returns:
        np.ndarray: The final cluster centroids (anchors) of shape (k, 2).
    """
    num_boxes = boxes_wh.shape[0]

    # Initialize centroids by randomly selecting k boxes from the dataset
    centroids = boxes_wh[np.random.choice(num_boxes, k, replace=False)]

    print(f"Running K-Means for {k} anchors...")
    for _ in tqdm(range(max_iters)):
        # Assign each box to the closest centroid (highest IoU)
        assignments = np.array([np.argmax(iou_numpy(box, centroids)) for box in boxes_wh])

        # Store old centroids to check for convergence
        old_centroids = centroids.copy()

        # Update centroids by calculating the median of all boxes in each cluster
        for i in range(k):
            cluster_boxes = boxes_wh[assignments == i]
            if len(cluster_boxes) > 0:
                centroids[i] = np.median(cluster_boxes, axis=0)

        # Check if centroids have converged
        if np.all(centroids == old_centroids):
            print("✅ K-Means converged.")
            break

    return centroids


def load_dimensions_from_labels(labels_dir, image_size):
    """
    Loads all bounding box dimensions (width, height) from YOLO format label files.
    Args:
        labels_dir (str): Path to the directory containing .txt label files.
        image_size (int): The size of the input images (e.g., 640).
    Returns:
        np.ndarray: A numpy array of shape (n, 2) with dimensions in pixels.
    """
    label_files = glob(os.path.join(labels_dir, "*.txt"))
    all_dims = []
    print(f"🔍 Found {len(label_files)} label files. Reading dimensions...")
    for file_path in tqdm(label_files):
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # parts[3] is normalized width, parts[4] is normalized height
                    w = float(parts[3]) * image_size
                    h = float(parts[4]) * image_size
                    all_dims.append([w, h])
    return np.array(all_dims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Optimal YOLO Anchors using K-Means")
    parser.add_argument("--labels_dir", type=str, default="./datasets/fire-smoke/train/labels",
                        help="Path to the directory with training label files.")
    parser.add_argument("--num_anchors", type=int, default=3,
                        help="The number of anchors to generate.")
    parser.add_argument("--image_size", type=int, default=640,
                        help="The input image size used during training.")
    args = parser.parse_args()

    # 1. Load all bounding box dimensions from your training set
    box_dimensions = load_dimensions_from_labels(args.labels_dir, args.image_size)

    if len(box_dimensions) == 0:
        print("❌ ERROR: No bounding boxes found. Check the labels directory path.")
    else:
        # 2. Run K-Means to find the optimal anchors
        anchors = kmeans(box_dimensions, args.num_anchors)

        # 3. Sort anchors by area (width * height)
        anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

        # 4. Calculate and print the average IoU
        avg_iou = np.mean([np.max(iou_numpy(box, anchors)) for box in box_dimensions])

        print("\n" + "=" * 50)
        print("🎉 K-MEANS ANCHOR CALCULATION COMPLETE 🎉")
        print("=" * 50)
        print(f"\nAverage IoU of all boxes with best possible anchor: {avg_iou:.4f}\n")
        print("Copy the following line into your training script:")

        # Format the output for easy copy-pasting
        anchor_string = "ANCHORS = [" + ", ".join([f"({int(w)},{int(h)})" for w, h in anchors]) + "]"
        print("\033[92m" + anchor_string + "\033[0m")  # Print in green color
        print("\n" + "=" * 50)