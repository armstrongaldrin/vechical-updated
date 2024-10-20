import ssl
import os
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import psutil
import gc

ssl._create_default_https_context = ssl._create_unverified_context
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img_dir = '/Users/balajia/Desktop/untitled folder 2/IDD_Detection/JPEGImages'
output_dir = '/Users/balajia/Desktop/untitled folder 2/data/output_images'
os.makedirs(output_dir, exist_ok=True)

def find_image_files(directory, extensions=('.jpg', '.jpeg', '.png')):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def plot_results(image, result, save_path):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    for box in result:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) in [2, 3, 5, 7]:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            box_w, box_h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, label, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.savefig(save_path)
    plt.close(fig)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f'Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB')

img_files = find_image_files(img_dir)
num_images_to_process = 500

img_files = img_files[:num_images_to_process]

for i, img_path in enumerate(img_files):
    print(f"Processing image {i + 1}/{num_images_to_process}: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Image at {img_path} could not be loaded.")
        continue
    img_rgb = img[..., ::-1] 
    results = model([img_rgb], size=640)
    save_name = os.path.splitext(os.path.basename(img_path))[0] + '_result.jpg'
    save_path = os.path.join(output_dir, save_name)
    plot_results(img_rgb, results.xyxy[0].numpy(), save_path)
    print_memory_usage()
    del img, img_rgb
    gc.collect()
print("Processing complete.")
