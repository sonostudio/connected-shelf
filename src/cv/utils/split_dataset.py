import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
SOURCE_FOLDER = "connected-shelf-dataset/obj_train_data"  # Your unzipped CVAT folder
DEST_FOLDER = "dataset_split"
TRAIN_RATIO = 0.8


# ---------------------

def setup_dirs(base_path):
    for split in ['train', 'val']:
        for content in ['images', 'labels']:
            os.makedirs(os.path.join(base_path, content, split), exist_ok=True)


def smart_split():
    source = Path(SOURCE_FOLDER)
    dest = Path(DEST_FOLDER)

    print(f"🕵️‍♂️ Debugging path: {source.absolute()}")
    if not source.exists():
        print("❌ Error: SOURCE_FOLDER does not exist! Check the folder name.")
        return

    # dictionary: { class_id: [list_of_tuples] }
    class_buckets = defaultdict(list)

    print("🔍 Scanning filenames (v2)...")

    # 1. Find all images first (case insensitive check)
    all_images = []
    # Force a recursive search for all common image types
    for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]:
        all_images.extend(source.rglob(ext))

    if not all_images:
        print("❌ No images found! listing contents of source folder:")
        print(list(source.glob("*")))
        return
    else:
        print(f"✅ Found {len(all_images)} images. Checking match pairs...")

    matched_count = 0

    for img_path in all_images:
        # 2. Extract Class ID from filename
        # Format: {class_id}_2026... -> "0_2026..." -> "0"
        try:
            class_id = img_path.name.split('_')[0]
        except IndexError:
            continue

        # 3. Find corresponding label file
        # We try 3 different places where CVAT might hide the label
        possible_labels = [
            img_path.with_suffix(".txt"),  # Same folder
            img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name,  # Parallel 'labels' folder
            Path(str(img_path.parent).replace("images", "labels")) / img_path.with_suffix(".txt").name
            # Common substitutions
        ]

        found_label = None
        for p in possible_labels:
            if p.exists():
                found_label = p
                break

        if not found_label:
            # DEBUG: Print the first failure to help troubleshoot
            if matched_count == 0:
                print(f"⚠️ Debug Failure: Found image '{img_path.name}' but could not find label.")
                print(f"   Checked locations: {[str(p) for p in possible_labels]}")
            continue

        matched_count += 1
        class_buckets[class_id].append((img_path, found_label))

    # 4. Split each bucket independently
    train_set = []
    val_set = []

    print(f"\n📊 Class Distribution:")
    sorted_classes = sorted(class_buckets.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for class_id in sorted_classes:
        files = class_buckets[class_id]
        random.shuffle(files)

        count = len(files)
        split_idx = int(count * TRAIN_RATIO)

        if count > 0 and split_idx == 0:
            split_idx = 1

        t_files = files[:split_idx]
        v_files = files[split_idx:]

        train_set.extend(t_files)
        val_set.extend(v_files)

        print(f"   Class ID {class_id}: {count} images -> {len(t_files)} Train / {len(v_files)} Val")

    if matched_count == 0:
        print("\n❌ Found images, but NO matching labels found. Check your folder structure!")
        return

    # 5. Move the files
    print(f"\n📦 Moving {len(train_set)} files to Train and {len(val_set)} to Val...")
    setup_dirs(dest)

    def move_batch(file_list, split_name):
        for img_path, lbl_path in file_list:
            shutil.copy2(img_path, dest / "images" / split_name / img_path.name)
            shutil.copy2(lbl_path, dest / "labels" / split_name / lbl_path.name)

    move_batch(train_set, 'train')
    move_batch(val_set, 'val')

    print(f"\n✅ Stratified Split Complete! Output: {DEST_FOLDER}")


if __name__ == "__main__":
    smart_split()