"""VisionPort Session 5 - COCO dataset exploration.

Run from the project root: python explore_coco.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


# Implementation A - Read and explore the COCO JSON
project_dir = Path(__file__).resolve().parent
json_path = project_dir / "data" / "idd" / "annotations" / "instances.json"

with open(json_path, "r", encoding="utf-8") as file:
    coco = json.load(file)

print("COCO JSON loaded successfully!")

images = coco["images"]
categories = coco["categories"]
annotations = coco["annotations"]

print("\nDATASET SUMMARY")
print("Images:", len(images))
print("Categories:", len(categories))
print("Annotations:", len(annotations))

print("\nAVAILABLE CATEGORIES")
for category in categories:
    print(category["id"], ":", category["name"])

print("\nIMAGE INFORMATION")
for image in images:
    print(
        "Image ID:", image["id"],
        "| File:", image["file_name"],
        "| Width:", image["width"],
        "| Height:", image["height"],
    )

print("\nOBJECT ANNOTATIONS")
for annotation in annotations:
    print(
        "Image ID:", annotation["image_id"],
        "| Category ID:", annotation["category_id"],
        "| Bounding Box:", annotation["bbox"],
    )


# Implementation B - Visualise bounding boxes
image_dir = project_dir / "data" / "idd" / "images"
output_dir = project_dir / "docs"
output_dir.mkdir(parents=True, exist_ok=True)

# None means the first image that exists and has annotations.
TARGET_IMAGE_ID = None

category_names = {
    category["id"]: category["name"]
    for category in categories
}

annotations_by_image = defaultdict(list)
for annotation in annotations:
    annotations_by_image[annotation["image_id"]].append(annotation)

selected_image = None
for image_info in images:
    if TARGET_IMAGE_ID is not None and image_info["id"] != TARGET_IMAGE_ID:
        continue
    candidate = image_dir / image_info["file_name"]
    if candidate.exists() and annotations_by_image[image_info["id"]]:
        selected_image = image_info
        break

print("\nDRAWING")
if selected_image is None:
    print("No matching annotated image found. Check image names and paths.")
else:
    image_path = image_dir / selected_image["file_name"]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for annotation in annotations_by_image[selected_image["id"]]:
        x, y, width, height = annotation["bbox"]
        x1, y1 = x, y
        x2, y2 = x + width, y + height

        label = category_names.get(annotation["category_id"], "Unknown")
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label_position = (x1, max(0, y1 - 15))
        draw.rectangle(draw.textbbox(label_position, label), fill="red")
        draw.text(label_position, label, fill="white")

    output_path = output_dir / f"coco_preview_{selected_image['id']}.jpg"
    image.save(output_path)
    print("Preview saved to:", output_path)


# Implementation C - Count objects per category and verify the data
print("\nOBJECTS PER CATEGORY")
category_counts = Counter(
    annotation["category_id"] for annotation in annotations
)
for category_id, name in category_names.items():
    print(
        f"{category_id:>3} : {name:<12} "
        f"{category_counts.get(category_id, 0)}"
    )

print(
    "\nImages with annotations   :",
    sum(1 for image in images if annotations_by_image.get(image["id"])),
)
print(
    "Images without annotations:",
    sum(1 for image in images if not annotations_by_image.get(image["id"])),
)

image_by_id = {image["id"]: image for image in images}

bad_image = [
    annotation["id"]
    for annotation in annotations
    if annotation["image_id"] not in image_by_id
]
bad_category = [
    annotation["id"]
    for annotation in annotations
    if annotation["category_id"] not in category_names
]
bad_size = [
    annotation["id"]
    for annotation in annotations
    if annotation["bbox"][2] <= 0 or annotation["bbox"][3] <= 0
]
outside = [
    annotation["id"]
    for annotation in annotations
    if annotation["image_id"] in image_by_id
    and (
        annotation["bbox"][0] < 0
        or annotation["bbox"][1] < 0
        or annotation["bbox"][0] + annotation["bbox"][2]
        > image_by_id[annotation["image_id"]]["width"]
        or annotation["bbox"][1] + annotation["bbox"][3]
        > image_by_id[annotation["image_id"]]["height"]
    )
]

print("\nVERIFICATION CHECKS")
checks = [
    ("Every annotation points to an existing image", bad_image),
    ("Every annotation points to an existing category", bad_category),
    ("Every box has width > 0 and height > 0", bad_size),
    ("Every box lies inside its image", outside),
]
for check_name, bad_ids in checks:
    result = "PASS" if not bad_ids else f"FAIL {bad_ids[:5]}"
    print(f"{check_name:<50} {result}")
