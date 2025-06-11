import os
import csv

labels_folder = os.path.join("data","raw","train","labels")
output_csv = os.path.join("data", "processed", "train_labels_ids.csv")
label_ids = []

for filename in os.listdir(labels_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(labels_folder,filename)
        with open(file_path,'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    label_ids.append([parts[0]])

# .strip(): removes any leading or trailing spaces or newline characters (\n).
# .split(): splits the line by whitespace into a list of strings

with open(output_csv,'w',newline = '')as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ClassID"]) 
    writer.writerows(label_ids)

print(f"Saved {len(label_ids)} entries to {output_csv}")


