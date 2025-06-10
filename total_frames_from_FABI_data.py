from pathlib import Path


ROOTS = [
    Path(r"C:\Users\hc1321\OneDrive - Imperial College London\4th year\Individual Project\Fabi Data\DATASETS\test_upsampled"),    
    Path(r"C:\Users\hc1321\OneDrive - Imperial College London\4th year\Individual Project\Fabi Data\DATASETS\train_upsampled"),  
]


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

grand_total = 0

for root in ROOTS:
    if not root.exists():
        print(f"⚠️  {root} does not exist – skipping\n")
        continue

    root_count = sum(
        1
        for f in root.rglob("*")       
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    grand_total += root_count
    print(f"{root}  →  {root_count:,} images")

print("-" * 60)
print(f"TOTAL images across all roots:  {grand_total:,}")
