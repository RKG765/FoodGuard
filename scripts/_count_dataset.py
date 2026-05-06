"""Quick dataset counter."""
from pathlib import Path

root = Path("e:/BML/Semester-VI/Prj-3")
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("=== ai_generated ===")
ai_dir = root / "ai_generated"
for d in sorted(ai_dir.iterdir()):
    if d.is_dir():
        count = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in exts)
        print(f"  {d.name}: {count}")

print("\n=== dataset_4class ===")
ds_dir = root / "dataset_4class"
for s in sorted(ds_dir.iterdir()):
    if s.is_dir():
        for c in sorted(s.iterdir()):
            if c.is_dir():
                count = sum(1 for f in c.rglob("*") if f.is_file())
                print(f"  {s.name}/{c.name}: {count}")

print("\n=== Real source pools ===")
real_dirs = {
    "food_101": root / "food_101" / "food-101" / "food-101" / "images",
    "food_image_dataset": root / "food_image_dataset",
    "indian_food_data": root / "indian_food_data",
}
total_real = 0
for name, d in real_dirs.items():
    if d.exists():
        count = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in exts)
        print(f"  {name}: {count}")
        total_real += count
    else:
        print(f"  {name}: NOT FOUND")
print(f"  TOTAL REAL POOL: {total_real}")
