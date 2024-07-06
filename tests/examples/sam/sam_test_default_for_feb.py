import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Define the coordinates corresponding to each image
coordinates = [
    (1523, 1711), (1328, 962), (653, 778), (1357, 424), (40, 263),
    (521, 933), (1196, 1603), (355, 750), (876, 200), (2347, 1076),
    (349, 555), (767, 1311), (893, 509), (653, 704), (1620, 1654),
    (1288, 658), (1814, 1099), (2478, 618), (1940, 492), (807, 149),
    (2404, 1150), (653, 361), (1757, 1654), (1208, 1294), (2003, 1339),
    (1316, 355), (893, 1168), (1042, 1185), (1837, 922), (435, 1586),
    (1385, 212), (538, 1276), (1105, 790), (927, 824), (1780, 847),
    (1580, 647), (2330, 114), (2032, 1076), (973, 727), (2587, 1505),
    (2118, 773), (1213, 1162), (1362, 1196), (418, 172), (52, 796),
    (1792, 1213), (406, 1030), (641, 1448), (595, 1076), (584, 1700),
    (549, 1723), (343, 172), (1110, 1608), (1122, 1133), (670, 1408),
    (1070, 509), (521, 1128), (326, 944), (1929, 675), (1242, 973),
    (538, 1013), (298, 1168), (1660, 1740), (2003, 189), (944, 1265),
    (2433, 458), (1637, 899), (395, 784), (979, 1505), (595, 1019),
    (2478, 1414), (1517, 1213), (979, 956), (1568, 675), (1173, 469),
    (1969, 1213), (767, 229), (2021, 338), (681, 1454), (1357, 1631),
    (1265, 321), (286, 538), (1477, 412), (1156, 1362), (2078, 1162),
    (1168, 1030), (2083, 1654), (1677, 973), (401, 1568), (1345, 859),
    (309, 1122), (887, 57), (1746, 990), (1391, 859), (1368, 538),
    (1128, 1076), (1746, 887), (527, 813), (2124, 1431), (1408, 979),
    (2158, 544), (1574, 46), (1007, 1505), (2043, 635), (401, 1460),
    (1099, 389), (2312, 538), (647, 1248), (114, 229), (1437, 1568),
    (790, 1540), (1374, 1374), (1620, 1219), (2633, 183), (1545, 1299),
    (1231, 1231), (1431, 1225), (126, 1231), (492, 1105), (292, 1242),
    (2478, 1334), (2559, 1580), (1345, 1683), (1305, 1311), (544, 1729),
    (1740, 1666), (2164, 973), (1797, 298), (52, 189), (200, 618),
    (498, 321), (664, 1420), (2244, 962), (756, 1156), (916, 378),
    (910, 544), (355, 653), (1357, 1660), (74, 824), (927, 1528),
    (1145, 1128), (446, 487), (527, 1631), (1053, 1179), (2438, 1173),
    (1408, 870), (853, 383), (1683, 200), (2055, 979), (252, 504),
    (1374, 1351), (1935, 280), (715, 830), (1814, 1196), (950, 778),
    (1362, 252), (653, 1385), (990, 904), (2404, 664), (401, 1110),
    (1482, 675), (1150, 1442), (1116, 1345), (973, 412), (1557, 1774),
    (57, 967), (1357, 1597), (1282, 1717), (406, 1322), (1952, 1591),
    (1792, 1431), (114, 996), (435, 1271), (1969, 235), (481, 899),
    (1986, 1362), (183, 1660), (979, 1133), (492, 967), (1826, 612),
    (1322, 1534), (2072, 1133), (206, 1568), (922, 1311), (1917, 1763),
    (967, 1471), (967, 1036), (1076, 1357), (796, 1265), (1568, 1168),
    (1059, 1397), (916, 1379), (544, 1242), (1002, 893), (612, 899),
    (1362, 773), (2456, 315), (200, 630), (1614, 1093), (2146, 57),
    (1843, 86), (956, 1145), (2066, 1603), (1168, 1088), (1385, 1076),
    (1975, 1574), (1030, 1259), (303, 1282), (2192, 641), (1328, 790),
    (1477, 985), (1637, 1517), (292, 927), (1019, 235), (767, 1551),
    (2501, 578), (1666, 1471), (1030, 1528), (2427, 86), (1133, 1397)
]

# Update the image directory and filenames
image_dir = '/home/minjilee/erc_tree_semantic_segmentation_in_mlops/seasonaldata/seasonaldata/feb/temp'
image_filenames = [f'image_{i:03d}.jpg' for i in range(1, 221)]

# Set up model
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/home/minjilee/erc_tree_semantic_segmentation_in_mlops/weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# Create output directory if it doesn't exist
output_dir = 'segmentation_results'
os.makedirs(output_dir, exist_ok=True)

# Iterate through the images and corresponding coordinates
for idx, (filename, (x, y)) in enumerate(zip(image_filenames, coordinates)):
    image_path = os.path.join(image_dir, filename)
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping.")
        continue
    
    # Load and display the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    input_point = np.array([[x, y]])
    input_label = np.array([1])

    # Perform prediction
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Save results
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_path = os.path.join(output_dir, f'mask_{idx+1}_seg_{i+1}.png')
        score_path = os.path.join(output_dir, f'score_{idx+1}_seg_{i+1}.txt')

        # Save the mask image
        mask_image = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(mask_path, mask_image)

        # Save the score
        with open(score_path, 'w') as f:
            f.write(f'Score: {score:.3f}\n')

    # Plot results
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Original image with input point
    axs[0].imshow(image)
    show_points(input_point, input_label, axs[0])
    axs[0].set_title(f'Input Point {idx+1}')
    axs[0].axis('off')

    # Masks with scores
    for i in range(3):  # Show up to 3 masks
        if i < len(masks):
            axs[i+1].imshow(image)
            show_mask(masks[i], axs[i+1])
            show_points(input_point, input_label, axs[i+1])
            axs[i+1].set_title(f'Mask {i+1} Score: {scores[i]:.3f}')
            axs[i+1].axis('off')

    # Save plot
    plot_path = os.path.join(output_dir, f'results_{idx+1}.png')
    plt.savefig(plot_path)
    plt.close()

print("Segmentation completed and results saved.")