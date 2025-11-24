# Notes on Dimension Analyzer

The **variation/tolerance percentage** in `dlcv_p2_analyze_dimensions.py` controls how images with slightly different dimensions get grouped together into the same bin.

When we run `python dlcv_p2_analyze_dimensions.py 15`, the `15` means **15% tolerance**.

## How Binning Works

The algorithm processes images sequentially (sorted by area) and assigns each unassigned image to a bin:

1. The first unassigned image becomes a **bin center**
2. All other **unassigned** images within tolerance range join that bin
3. Assigned images are marked as "used" and cannot join other bins
4. Process continues with the next unassigned image

## Example with 15% Tolerance

Given these images sorted by area:
- Image A: 900×750 (area: 675,000)
- Image B: 1000×800 (area: 800,000)  
- Image C: 1050×850 (area: 892,500)
- Image D: 1100×700 (area: 770,000)
- Image E: 1200×800 (area: 960,000)

**Step 1**: Process Image A (900×750)
- Creates **Bin 1** centered at 900×750
- Tolerance ranges: 765-1035 width, 638-863 height
- ✅ Image B (1000×800) → **joins Bin 1** (within range)
- ✅ Image C (1050×850) → **joins Bin 1** (within range)
- ❌ Image D (1100×700) → separate bin (700 < 638, outside height range)
- ❌ Image E (1200×800) → separate bin (1200 > 1035, outside width range)
- **Bin 1**: {900×750, 1000×800, 1050×850}

**Step 2**: Process Image D (1100×700) - next unassigned
- Creates **Bin 2** centered at 1100×700
- Tolerance ranges: 935-1265 width, 595-805 height
- ✅ Image E (1200×800) → **joins Bin 2** (within range)
- **Bin 2**: {1100×700, 1200×800}

**Final Result**: 2 bins created

> **IMPORTANT**: The tolerance defines which images are **compatible** for grouping, but the **processing order** (by area) determines the actual bin assignments. Once an image is assigned to a bin, it cannot join another bin, even if it would be within tolerance range of that bin's center.

## Why This Matters

1. **Fewer bins** (higher tolerance like 15-30%):
   - Groups more images together
   - Fewer dimension configurations to manage
   - More aggressive normalization (might lose some detail)
   - Faster to configure

2. **More bins** (lower tolerance like 5-10%):
   - More precise dimension matching
   - More configurations to manage
   - Better preserves original image characteristics
   - Takes longer to configure all dimensions

## Practical Impact

- `15%` tolerance → ~68 bins for the dataset
- `10%` tolerance → ~116 bins
- `5%` tolerance → Could be 200+ bins

---