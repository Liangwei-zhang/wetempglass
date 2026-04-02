#!/usr/bin/env python3
"""
專業級抠圖腳本 — 移除棋盤格背景，只保留金色文字 + 金屬結構，背景完全透明。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# ── 可調常數 ──────────────────────────────────────────────────────
CHECKER_PATTERN_MIN_DIFF = 5.0
EDGE_ALPHA_MIN_DIST = 15.0
EDGE_ALPHA_RANGE = 40.0
AUTO_THRESHOLD_CLOSE_BG_DIST = 20.0
AUTO_THRESHOLD_FAR_BG_DIST = 50.0
AUTO_THRESHOLD_ADJUSTMENT = 10.0


def kmeans2_simple(pixels: np.ndarray, max_iter: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pixels), size=2, replace=False)
    centers = pixels[idx].astype(np.float64)

    for _ in range(max_iter):
        dist0 = np.linalg.norm(pixels.astype(np.float64) - centers[0], axis=1)
        dist1 = np.linalg.norm(pixels.astype(np.float64) - centers[1], axis=1)
        labels = (dist1 < dist0).astype(int)
        new_c0 = pixels[labels == 0].mean(axis=0) if (labels == 0).any() else centers[0]
        new_c1 = pixels[labels == 1].mean(axis=0) if (labels == 1).any() else centers[1]
        if np.allclose(centers, [new_c0, new_c1], atol=0.5):
            break
        centers = np.array([new_c0, new_c1])

    return centers[0].astype(np.uint8), centers[1].astype(np.uint8)


def detect_checker_colors_from_corners(img_arr: np.ndarray, sample_size: int = 30) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_arr.shape[:2]
    corners = [
        img_arr[:sample_size, :sample_size],
        img_arr[:sample_size, w - sample_size:],
        img_arr[h - sample_size:, :sample_size],
        img_arr[h - sample_size:, w - sample_size:],
    ]
    samples = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return kmeans2_simple(samples)


def color_distance(img: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((img.astype(np.float64) - target.astype(np.float64)) ** 2, axis=2))


def build_foreground_mask(img_arr: np.ndarray, color_a: np.ndarray, color_b: np.ndarray, threshold: float = 35.0) -> np.ndarray:
    dist_a = color_distance(img_arr, color_a)
    dist_b = color_distance(img_arr, color_b)
    min_dist = np.minimum(dist_a, dist_b)
    mask = (min_dist > threshold).astype(np.uint8) * 255
    return mask


def refine_mask(mask: np.ndarray) -> np.ndarray:
    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.filter(ImageFilter.MedianFilter(size=3))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=3))
    mask_img = mask_img.filter(ImageFilter.MinFilter(size=3))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return np.array(mask_img)


def enhance_edge_alpha(img_arr: np.ndarray, mask: np.ndarray, color_a: np.ndarray, color_b: np.ndarray) -> np.ndarray:
    dist_a = color_distance(img_arr, color_a)
    dist_b = color_distance(img_arr, color_b)
    min_bg_dist = np.minimum(dist_a, dist_b)

    edge_zone = (mask > 20) & (mask < 235)
    refined = mask.copy().astype(np.float64)

    if edge_zone.any():
        edge_alpha = np.clip((min_bg_dist - EDGE_ALPHA_MIN_DIST) / EDGE_ALPHA_RANGE, 0.0, 1.0) * 255
        refined[edge_zone] = edge_alpha[edge_zone]

    return np.clip(refined, 0, 255).astype(np.uint8)


def detect_checker_pattern(img_arr: np.ndarray, block_size: int = 16) -> bool:
    h, w = img_arr.shape[:2]
    region = img_arr[:min(64, h), :min(64, w)]
    if region.size == 0:
        return False

    bs = block_size
    diffs = []
    for y in range(0, min(64, h) - bs, bs):
        for x in range(0, min(64, w) - bs, bs):
            block = region[y:y+bs, x:x+bs].mean(axis=(0, 1))
            if x + bs < min(64, w):
                right_block = region[y:y+bs, x+bs:x+2*bs].mean(axis=(0, 1))
                diffs.append(np.linalg.norm(block - right_block))

    if not diffs:
        return False

    avg_diff = np.mean(diffs)
    return avg_diff > CHECKER_PATTERN_MIN_DIFF


def process_image(input_path: str, output_path: str | None = None, threshold: float = 35.0, auto_threshold: bool = True) -> str:
    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"找不到輸入檔案: {input_path}")

    if output_path is None:
        output_path = str(input_p.with_name(f"{input_p.stem}_cutout.png"))

    print(f"📂 載入圖片: {input_path}")
    img = Image.open(input_path).convert("RGB")
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]
    print(f" 尺寸: {w} × {h}")

    print("🔍 偵測棋盤格背景色...")
    color_a, color_b = detect_checker_colors_from_corners(img_arr, sample_size=30)
    print(f" 背景色 A: RGB({color_a[0]}, {color_a[1]}, {color_a[2]})")
    print(f" 背景色 B: RGB({color_b[0]}, {color_b[1]}, {color_b[2]})")

    bg_dist = np.linalg.norm(color_a.astype(float) - color_b.astype(float))
    print(f" 兩色差距: {bg_dist:.1f}")

    if not detect_checker_pattern(img_arr):
        print("⚠️ 未偵測到明顯棋盤格圖案，嘗試以邊角平均色作為背景色處理。")

    if auto_threshold:
        if bg_dist < AUTO_THRESHOLD_CLOSE_BG_DIST:
            threshold = max(25.0, threshold - AUTO_THRESHOLD_ADJUSTMENT)
        elif bg_dist > AUTO_THRESHOLD_FAR_BG_DIST:
            threshold = min(50.0, threshold + AUTO_THRESHOLD_ADJUSTMENT / 2)
        print(f" 自動閾值: {threshold:.1f}")

    print("🎭 建立前景遮罩...")
    mask = build_foreground_mask(img_arr, color_a, color_b, threshold=threshold)

    print("✨ 形態學精修遮罩...")
    mask = refine_mask(mask)

    print("🖌️ 邊緣 alpha 精緻化...")
    mask = enhance_edge_alpha(img_arr, mask, color_a, color_b)

    print("📦 組合 RGBA 輸出...")
    rgba = np.dstack([img_arr, mask])
    result = Image.fromarray(rgba, mode="RGBA")

    result.save(output_path, format="PNG", optimize=True)
    print(f"✅ 已儲存至: {output_path}")

    total_px = h * w
    fg_px = int((mask > 128).sum())
    print(f"📊 前景像素: {fg_px:,} / {total_px:,} ({fg_px / total_px * 100:.1f}%)")

    return output_path


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    threshold = 35.0
    for i, arg in enumerate(sys.argv):
        if arg in ("--threshold", "-t") and i + 1 < len(sys.argv):
            try:
                threshold = float(sys.argv[i + 1])
            except ValueError:
                print(f"❌ 閾值參數必須是數字，收到: {sys.argv[i + 1]}")
                sys.exit(1)

    process_image(input_path, output_path, threshold=threshold)


if __name__ == "__main__":
    main()