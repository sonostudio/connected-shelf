import yaml
import os

# Read class names from data.yaml (update the path as needed)
with open('src/cv/data/project-name-version/data.yaml', 'r') as f:
    data = yaml.safe_load(f)

class_names = data['names']

# Create labels file
os.makedirs("src/cv/models", exist_ok=True)
with open('src/cv/models/labels.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print(f"✓ Labels saved: src/cv/models/labels.txt")
print(f"Classes: {', '.join(class_names)}")
