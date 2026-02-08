"""
Build dataset_index.csv from all food datasets.

Creates unified CSV with columns:
- image_path: relative path to image
- source_dataset: food101, indian_food, uecfood256, aircrowd
- category: one of [indian, fast_food, street_food, desserts, beverages, continental]
- label: 'real' for all images (AI images added later)

Usage:
    python scripts/build_csv.py
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT


def load_category_mapping() -> Dict:
    """Load category mapping from YAML."""
    mapping_path = PROJECT_ROOT / 'config' / 'category_mapping.yaml'
    with open(mapping_path, 'r') as f:
        return yaml.safe_load(f)


def get_category_from_keywords(name: str, keyword_map: Dict[str, str], default: str = 'continental') -> str:
    """Match category based on keywords in name."""
    name_lower = name.lower()
    for keyword, category in keyword_map.items():
        if keyword != 'default' and keyword in name_lower:
            return category
    return keyword_map.get('default', default)


def process_food101(mapping: Dict) -> List[Tuple[str, str, str, str]]:
    """Process Food-101 dataset."""
    rows = []
    base_path = DATA_ROOT / 'food_101' / 'food-101' / 'food-101'
    images_path = base_path / 'images'
    classes_file = base_path / 'meta' / 'classes.txt'
    
    if not classes_file.exists():
        print(f"Warning: Food-101 classes.txt not found at {classes_file}")
        return rows
    
    # Load class names
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    food101_map = mapping.get('food_101', {})
    
    for class_name in classes:
        class_dir = images_path / class_name
        if not class_dir.exists():
            continue
        
        # Get category from mapping, default to continental
        category = food101_map.get(class_name, 'continental')
        
        for img_file in class_dir.glob('*.jpg'):
            rel_path = img_file.relative_to(DATA_ROOT)
            rows.append((str(rel_path), 'food101', category, 'real'))
    
    return rows


def process_indian_food(mapping: Dict) -> List[Tuple[str, str, str, str]]:
    """Process Indian Food dataset."""
    rows = []
    # Try multiple possible paths
    possible_paths = [
        DATA_ROOT / 'indian_food_data' / 'image_for _cuisines' / 'data',
        DATA_ROOT / 'indian_food_data' / 'image_for_cuisines' / 'data',
        DATA_ROOT / 'indian_food_data' / 'images',
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
    
    if not data_path:
        print(f"Warning: Indian food data not found")
        return rows
    
    for img_file in data_path.glob('*.jpg'):
        rel_path = img_file.relative_to(DATA_ROOT)
        rows.append((str(rel_path), 'indian_food', 'indian', 'real'))
    
    # Also check for PNG
    for img_file in data_path.glob('*.png'):
        rel_path = img_file.relative_to(DATA_ROOT)
        rows.append((str(rel_path), 'indian_food', 'indian', 'real'))
    
    return rows


def process_uecfood256(mapping: Dict) -> List[Tuple[str, str, str, str]]:
    """Process UECFOOD256 dataset."""
    rows = []
    base_path = DATA_ROOT / 'UECFOOD256'
    
    if not base_path.exists():
        print(f"Warning: UECFOOD256 not found at {base_path}")
        return rows
    
    category_file = base_path / 'category.txt'
    category_names = {}
    
    if category_file.exists():
        with open(category_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0])
                        category_names[idx] = parts[1]
                    except ValueError:
                        pass
    
    uec_map = mapping.get('uecfood256', {})
    
    # Iterate through numbered folders
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            folder_id = int(folder.name)
            class_name = category_names.get(folder_id, f'class_{folder_id}')
            
            # Determine category from keywords
            category = get_category_from_keywords(class_name, uec_map)
            
            for img_file in folder.glob('*.jpg'):
                rel_path = img_file.relative_to(DATA_ROOT)
                rows.append((str(rel_path), 'uecfood256', category, 'real'))
    
    return rows


def process_aircrowd(mapping: Dict) -> List[Tuple[str, str, str, str]]:
    """Process Aircrowd Food Recognition dataset."""
    rows = []
    # Common Aircrowd paths
    possible_paths = [
        DATA_ROOT / 'aircrowd',
        DATA_ROOT / 'food_recognition',
        DATA_ROOT / 'aicrowd',
    ]
    
    base_path = None
    for p in possible_paths:
        if p.exists():
            base_path = p
            break
    
    if not base_path:
        print(f"Warning: Aircrowd dataset not found")
        return rows
    
    aircrowd_map = mapping.get('aircrowd', {})
    
    # Look for images in subdirectories
    for img_file in base_path.rglob('*.jpg'):
        # Get category from filename/path keywords
        name = img_file.stem + '_' + str(img_file.parent.name)
        category = get_category_from_keywords(name, aircrowd_map)
        
        rel_path = img_file.relative_to(DATA_ROOT)
        rows.append((str(rel_path), 'aircrowd', category, 'real'))
    
    return rows


def build_csv():
    """Build the unified dataset_index.csv."""
    print("Building dataset_index.csv...")
    print(f"Data root: {DATA_ROOT}")
    
    # Load category mapping
    mapping = load_category_mapping()
    
    all_rows = []
    
    # Process each dataset
    print("\nProcessing Food-101...")
    food101_rows = process_food101(mapping)
    print(f"  Found {len(food101_rows)} images")
    all_rows.extend(food101_rows)
    
    print("\nProcessing Indian Food...")
    indian_rows = process_indian_food(mapping)
    print(f"  Found {len(indian_rows)} images")
    all_rows.extend(indian_rows)
    
    print("\nProcessing UECFOOD256...")
    uec_rows = process_uecfood256(mapping)
    print(f"  Found {len(uec_rows)} images")
    all_rows.extend(uec_rows)
    
    print("\nProcessing Aircrowd...")
    aircrowd_rows = process_aircrowd(mapping)
    print(f"  Found {len(aircrowd_rows)} images")
    all_rows.extend(aircrowd_rows)
    
    # Write CSV
    output_path = DATA_ROOT / 'dataset_index.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'source_dataset', 'category', 'label'])
        writer.writerows(all_rows)
    
    print(f"\n{'='*50}")
    print(f"Total images: {len(all_rows)}")
    print(f"Output: {output_path}")
    
    # Category distribution
    print("\nCategory distribution:")
    categories = {}
    for row in all_rows:
        cat = row[2]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Source distribution
    print("\nSource distribution:")
    sources = {}
    for row in all_rows:
        src = row[1]
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")


if __name__ == '__main__':
    build_csv()
