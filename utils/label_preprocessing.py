import os
import csv
from image_preprocessing import count
import numpy as np

labels_folder = os.path.join("data", "raw", "train", "labels")
output_csv = os.path.join("data", "processed", "train_labels.csv")

class_names = [
    'Ascending-Triangle', 'Channel-down', 'Channel-up', 'Cup-and-handle', 'Descending-Triangle',
    'Double-Bottom', 'Double-Top', 'Falling-Wedge', 'Head-Shoulders', 'Inverse-Head-Shoulders',
    'Resistance-Emerging', 'Resistance-breakout', 'Rising-Wedge', 'Rounding-Bottom',
    'Rounding-Top', 'Support-breakout', 'Triangle', 'Triple-Bottom', 'Triple-Top', 'rectangle'
]

label_data = []
filename_id = 1

for filename in sorted(os.listdir(labels_folder)):
    if filename.endswith(".txt"):
        file_path = os.path.join(labels_folder, filename)
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_name = class_names[class_id]
                    label_data.append([filename_id, class_id, class_name])
        filename_id += 1 

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "ClassID", "ClassName"])
    writer.writerows(label_data)
