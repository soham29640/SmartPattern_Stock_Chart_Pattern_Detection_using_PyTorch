import os
import csv

label_dir = os.path.join("data", "raw", "train", "labels")
image_dir = os.path.join("data", "processed", "train_images")
output_csv = os.path.join("data", "processed", "train_labels.csv")

class_names = [
    'Ascending-Triangle', 'Channel-down', 'Channel-up', 'Cup-and-handle', 'Descending-Triangle',
    'Double-Bottom', 'Double-Top', 'Falling-Wedge', 'Head-Shoulders', 'Inverse-Head-Shoulders',
    'Resistance-Emerging', 'Resistance-breakout', 'Rising-Wedge', 'Rounding-Bottom',
    'Rounding-Top', 'Support-breakout', 'Triangle', 'Triple-Bottom', 'Triple-Top', 'rectangle'
]

label_data = []

original_images = sorted([
    f for f in os.listdir(os.path.join("data", "raw", "train", "images"))
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
])

for i, original_image in enumerate(original_images, start=1):
    base_name = os.path.splitext(original_image)[0]
    label_file = os.path.join(label_dir, base_name + ".txt")

    if os.path.exists(label_file):
        with open(label_file, 'r') as file:
            first_line = file.readline().strip()
            if first_line:
                class_id = int(first_line.split()[0])
                label_data.append([i, class_id, class_names[class_id]])
    else:
        print(f"[WARN] Missing label for: {original_image}")

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "ClassID", "ClassName"])
    writer.writerows(label_data)

print(f"[INFO] Written {len(label_data)} entries to {output_csv}")
