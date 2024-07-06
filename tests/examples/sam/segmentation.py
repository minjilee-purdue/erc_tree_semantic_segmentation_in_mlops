import os
import re
import cv2
import numpy as np
from typing import Tuple, List
import gradio as gr
from segment_anything import SamPredictor, sam_model_registry

# Define your checkpoint path
checkpoint = "/home/minjilee/Desktop/project_directory/weights/sam_vit_h_4b8939.pth"
predictor = SamPredictor(sam_model_registry["default"](checkpoint=checkpoint))

# Directories for annotations
directories = {
    "training": {
        "bbox_coords": "/home/minjilee/Desktop/seasonaldata/feb/training/processed/annotations_boundingbox_coords",
        "bbox": "/home/minjilee/Desktop/seasonaldata/feb/training/processed/annotations_boundingbox",
        "masks": "/home/minjilee/Desktop/seasonaldata/feb/training/processed/annotations_masks",
        "click_coords": "/home/minjilee/Desktop/seasonaldata/feb/training/processed/click_coords"
    },
    "testing": {
        "bbox_coords": "/home/minjilee/Desktop/seasonaldata/feb/testing/processed/annotations_boundingbox_coords",
        "bbox": "/home/minjilee/Desktop/seasonaldata/feb/testing/processed/annotations_boundingbox",
        "masks": "/home/minjilee/Desktop/seasonaldata/feb/testing/processed/annotations_masks",
        "click_coords": "/home/minjilee/Desktop/seasonaldata/feb/testing/processed/click_coords"
    },
    "evaluation": {
        "bbox_coords": "/home/minjilee/Desktop/seasonaldata/feb/evaluation/processed/annotations_boundingbox_coords",
        "bbox": "/home/minjilee/Desktop/seasonaldata/feb/evaluation/processed/annotations_boundingbox",
        "masks": "/home/minjilee/Desktop/seasonaldata/feb/evaluation/processed/annotations_masks",
        "click_coords": "/home/minjilee/Desktop/seasonaldata/feb/evaluation/processed/click_coords"
    }
}


# Ensure the directories exist
for category in directories.values():
    for path in category.values():
        os.makedirs(path, exist_ok=True)

def segment(image: np.ndarray, point_x: int, point_y: int) -> np.ndarray:
    """
    Segment the image based on a click point.

    Args:
        image (np.ndarray): The input image.
        point_x (int): X-coordinate of the click.
        point_y (int): Y-coordinate of the click.

    Returns:
        np.ndarray: The segmented mask in BGR format.
    """
    points_coords = np.array([[point_x, point_y], [0, 0]])
    points_label = np.array([1, -1])

    masks, scores, _ = predictor.predict(points_coords, points_label)
    mask, _ = select_masks(masks, scores, points_coords.shape[0])
    mask = (mask > 0).astype(np.uint8) * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return mask

def select_masks(masks: np.ndarray, iou_preds: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the best mask based on IOU predictions.

    Args:
        masks (np.ndarray): Array of masks.
        iou_preds (np.ndarray): Array of IOU predictions.
        num_points (int): Number of points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The best mask and its IOU prediction.
    """
    score_reweight = np.array([1000] + [0] * 2)
    score = iou_preds + (num_points - 2.5) * score_reweight
    best_idx = np.argmax(score)
    mask = np.expand_dims(masks[best_idx, :, :], axis=-1)
    iou_pred = np.expand_dims(iou_preds[best_idx], axis=0)

    return mask, iou_pred

def set_image(image: np.ndarray):
    """
    Preprocess image and get image embedding with SAM Encoder.

    Args:
        image (np.ndarray): The input image.
    """
    predictor.set_image(image)

def get_coords(evt: gr.SelectData) -> Tuple[int, int]:
    """
    Get the coordinates from the Gradio SelectData event.

    Args:
        evt (gr.SelectData): The event data.

    Returns:
        Tuple[int, int]: The x and y coordinates.
    """
    return evt.index[0], evt.index[1]

def get_next_file_number(directory: str, prefix: str, extension: str) -> int:
    """
    Get the next file number for saving results.

    Args:
        directory (str): The directory to look for files.
        prefix (str): The prefix of the files.
        extension (str): The extension of the files.

    Returns:
        int: The next file number.
    """
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    if not files:
        return 1
    numbers = [int(re.search(r'\d+', f).group()) for f in files]
    return max(numbers) + 1

def save_click_coordinates(click_w: int, click_h: int):
    """
    Save the click coordinates to a text file.

    Args:
        click_w (int): X-coordinate of the click.
        click_h (int): Y-coordinate of the click.
    """
    file_number = get_next_file_number(directories["testing"]["click_coords"], "click_coords_", ".txt")
    file_number_str = f"{file_number:03}"
    coords_file_path = os.path.join(directories["testing"]["click_coords"], f"click_coords_{file_number_str}.txt")

    with open(coords_file_path, "w") as file:
        file.write(f"Clicked coordinates: {click_w}, {click_h}\n")

def segment_by_click(image: np.ndarray, evt: gr.SelectData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment the image based on a click event.

    Args:
        image (np.ndarray): The input image.
        evt (gr.SelectData): The click event data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The segmented image and the image with bounding boxes.
    """
    click_w, click_h = evt.index
    print(f"Clicked coordinates: {click_w}, {click_h}")

    predictor.set_image(image)
    segmented_image = segment(image, click_w, click_h)

    print(f"Segmented image shape: {segmented_image.shape}")

    bounding_boxes = calculate_bounding_boxes(segmented_image)
    segmented_image_with_boxes = draw_bounding_boxes(segmented_image, bounding_boxes)

    # Save results including click coordinates
    save_results((click_w, click_h), bounding_boxes, segmented_image, segmented_image_with_boxes)

    return segmented_image, segmented_image_with_boxes

def calculate_bounding_boxes(segmented_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Calculate bounding boxes from the segmented image.

    Args:
        segmented_image (np.ndarray): The segmented image.

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding box coordinates.
    """
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, x + w, y, y + h))

    return bounding_boxes

def draw_bounding_boxes(image: np.ndarray, bounding_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Draw bounding boxes on the image.

    Args:
        image (np.ndarray): The input image.
        bounding_boxes (List[Tuple[int, int, int, int]]): List of bounding box coordinates.

    Returns:
        np.ndarray: The image with bounding boxes drawn.
    """
    image_with_boxes = image.copy()
    for bbox in bounding_boxes:
        x_min, x_max, y_min, y_max = bbox
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    return image_with_boxes

def save_results(click_coords: Tuple[int, int], bounding_boxes: List[Tuple[int, int, int, int]], segmented_image: np.ndarray, segmented_image_with_boxes: np.ndarray):
    """
    Save the results of segmentation, bounding boxes, and click coordinates.

    Args:
        click_coords (Tuple[int, int]): Coordinates of the click (x, y).
        bounding_boxes (List[Tuple[int, int, int, int]]): List of bounding box coordinates.
        segmented_image (np.ndarray): The segmented image.
        segmented_image_with_boxes (np.ndarray): The segmented image with bounding boxes drawn.
    """
    file_number = get_next_file_number(directories["training"]["click_coords"], "click_coords_", ".txt")
    file_number_str = f"{file_number:03}"

    click_coords_file_path = os.path.join(directories["training"]["click_coords"], f"click_coords_{file_number_str}.txt")
    bbox_file_path = os.path.join(directories["training"]["bbox_coords"], f"bbox_coords_{file_number_str}.txt")
    segmented_image_path = os.path.join(directories["training"]["masks"], f"segmented_image_{file_number_str}.png")
    segmented_image_with_boxes_path = os.path.join(directories["training"]["bbox"], f"segmented_image_with_boxes_{file_number_str}.png")

    # Save click coordinates
    with open(click_coords_file_path, "w") as file:
        file.write(f"Clicked coordinates: {click_coords[0]}, {click_coords[1]}\n")

    # Save bounding box coordinates
    with open(bbox_file_path, "w") as file:
        for bbox in bounding_boxes:
            file.write(f"[x_min: {bbox[0]}, x_max: {bbox[1]}], [y_min: {bbox[2]}, y_max: {bbox[3]}]\n")

    # Save segmented images
    cv2.imwrite(segmented_image_path, segmented_image)
    cv2.imwrite(segmented_image_with_boxes_path, segmented_image_with_boxes)
