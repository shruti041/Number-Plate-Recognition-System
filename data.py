import os
import shutil
import random

base_dir = "C:\\Users\\INSPIRON\\Desktop\\AI&ML project\\Number Plate Recognition System\\dataset"

images_dir = "C:\\Users\\INSPIRON\\Desktop\\AI&ML project\\Number Plate Recognition System\\dataset\\Indian_Number_Plates\\Sample_Images"
labels_dir = "C:\\Users\\INSPIRON\\Desktop\\AI&ML project\\Number Plate Recognition System\\dataset\\labels"

for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
print(f"Found {len(image_files)} images")

random.shuffle(image_files)
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_files(file_list, subset):
    for filename in file_list:
        print(f"Copying {filename} to {subset}")
        src_img = os.path.join(images_dir, filename)
        dst_img = os.path.join(base_dir, f'images/{subset}', filename)
        shutil.copy2(src_img, dst_img)

        label_name = os.path.splitext(filename)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(base_dir, f'labels/{subset}', label_name)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"⚠️ Label file missing for image: {filename}")

move_files(train_files, 'train')
move_files(val_files, 'val')

print("Dataset split done!")
