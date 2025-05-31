import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_path, yolo_label_path, class_id=0):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    converted = False  # Flag to check if anything was written

    with open(yolo_label_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text.lower()
            print(f"→ Found object: {cls} in {xml_path}")

            if cls != 'number_plate':  
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            box_width = xmax - xmin
            box_height = ymax - ymin

            x_center_norm = x_center / width
            y_center_norm = y_center / height
            width_norm = box_width / width
            height_norm = box_height / height

            out_file.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")
            converted = True

    return converted

annotations_dir = "C:\\Users\\INSPIRON\\Desktop\\AI&ML project\\Number Plate Recognition System\\dataset\\Annotations\\Annotations"
labels_dir = "C:\\Users\\INSPIRON\\Desktop\\AI&ML project\\Number Plate Recognition System\\dataset\\labels"

os.makedirs(labels_dir, exist_ok=True)

converted_count = 0
skipped_count = 0

for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith(".xml"):
        continue
    xml_path = os.path.join(annotations_dir, xml_file)
    label_file = os.path.splitext(xml_file)[0] + ".txt"
    yolo_label_path = os.path.join(labels_dir, label_file)

    if convert_voc_to_yolo(xml_path, yolo_label_path):
        print(f"✅ Saved: {yolo_label_path}")
        converted_count += 1
    else:
        print(f"⚠️ Skipped (no 'number_plate' class found): {xml_file}")
        skipped_count += 1

print(f"\nDone. {converted_count} labels saved. {skipped_count} files skipped.")
