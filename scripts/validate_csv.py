"""
Validate dataset_index.csv - check for null/corrupted images.

Removes invalid entries and creates a clean CSV.

Usage:
    python scripts/validate_csv.py
"""

import csv
from pathlib import Path
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT


def validate_image(row: tuple) -> tuple:
    """
    Validate a single image.
    Returns (row, is_valid, error_message)
    """
    image_path, source_dataset, category, label = row
    full_path = DATA_ROOT / image_path
    
    # Check if path is null/empty
    if not image_path or image_path.strip() == '':
        return row, False, "Empty path"
    
    # Check if file exists
    if not full_path.exists():
        return row, False, "File not found"
    
    # Check if file is readable and not corrupted
    try:
        with Image.open(full_path) as img:
            img.verify()  # Verify image integrity
        
        # Re-open to check if it can be loaded
        with Image.open(full_path) as img:
            img.load()  # Force load to detect truncated images
            
            # Check for valid dimensions
            if img.width < 10 or img.height < 10:
                return row, False, f"Invalid dimensions: {img.width}x{img.height}"
            
    except Exception as e:
        return row, False, f"Corrupted: {str(e)[:50]}"
    
    return row, True, None


def validate_csv():
    """Validate all images in dataset_index.csv."""
    csv_path = DATA_ROOT / 'dataset_index.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run build_csv.py first.")
        return
    
    print("Loading dataset_index.csv...")
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 4:
                rows.append(tuple(row[:4]))
    
    print(f"Total entries: {len(rows)}")
    print("\nValidating images (this may take a while)...")
    
    valid_rows = []
    invalid_rows = []
    
    # Use thread pool for faster validation
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(validate_image, row): row for row in rows}
        
        for future in tqdm(as_completed(futures), total=len(rows), desc="Validating"):
            row, is_valid, error = future.result()
            if is_valid:
                valid_rows.append(row)
            else:
                invalid_rows.append((row, error))
    
    # Report invalid images
    if invalid_rows:
        print(f"\n{'='*50}")
        print(f"Found {len(invalid_rows)} invalid images:")
        
        # Group by error type
        error_types = {}
        for row, error in invalid_rows:
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {error}: {count}")
        
        # Save invalid entries for review
        invalid_path = DATA_ROOT / 'invalid_images.csv'
        with open(invalid_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'source_dataset', 'category', 'label', 'error'])
            for row, error in invalid_rows:
                writer.writerow(list(row) + [error])
        print(f"\nInvalid entries saved to: {invalid_path}")
    
    # Check for duplicate paths
    print("\nChecking for duplicate paths...")
    seen_paths = set()
    unique_rows = []
    duplicate_count = 0
    for row in valid_rows:
        if row[0] not in seen_paths:
            seen_paths.add(row[0])
            unique_rows.append(row)
        else:
            duplicate_count += 1
    
    if duplicate_count > 0:
        print(f"  Removed {duplicate_count} duplicate paths")
    valid_rows = unique_rows
    
    # Write cleaned CSV
    clean_path = DATA_ROOT / 'dataset_index.csv'
    with open(clean_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(valid_rows)
    
    print(f"\n{'='*50}")
    print(f"Validation complete!")
    print(f"Valid images: {len(valid_rows)}")
    print(f"Invalid images: {len(invalid_rows)} (removed)")
    print(f"Duplicates: {duplicate_count} (removed)")
    print(f"Clean CSV saved to: {clean_path}")
    
    # Show category distribution after cleaning
    print("\nCategory distribution (after cleaning):")
    categories = {}
    for row in valid_rows:
        cat = row[2]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == '__main__':
    validate_csv()
