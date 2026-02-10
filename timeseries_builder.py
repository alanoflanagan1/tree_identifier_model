import os
from datetime import datetime
import cv2
import pandas as pd

from mask_selector import select_reference_mask, save_reference_outputs
from foliage_metric import foliage_metric


def extract_datetime(fname: str):
    try:
        parts = fname.split("_")
        date = parts[-2]
        time = parts[-1].split(".")[0]
        return datetime.strptime(date + time, "%Y%m%d%H%M%S")
    except Exception:
        return None


def build_timeseries_with_reference_masks(
    image_base: str,
    label_base: str,
    output_csv: str = "foliage_timeseries.csv",
    baseline_quantile: float = 0.85,
    reference_out_dir: str = "reference_masks",
):
    records = []

    # Iterate prediction folders (predict_<camera>_<hour>)
    for pred_folder in os.listdir(label_base):
        if not pred_folder.startswith("predict_"):
            continue

        base = pred_folder.replace("predict_", "", 1)
        if "_" not in base:
            continue

        camera, hour = base.rsplit("_", 1)

        label_dir = os.path.join(label_base, pred_folder, "labels")
        if not os.path.isdir(label_dir):
            continue

        img_dir = os.path.join(image_base, camera, hour)
        if not os.path.isdir(img_dir):
            print(f"⚠️ Missing image folder: {img_dir}")
            continue

        # --- Get a sample image to define shape ---
        sample_img_path = None
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_img_path = os.path.join(img_dir, fn)
                break

        if not sample_img_path:
            print(f"⚠️ No images found in: {img_dir}")
            continue

        sample_img = cv2.imread(sample_img_path)
        if sample_img is None:
            print(f"⚠️ Could not read sample image: {sample_img_path}")
            continue

        # --- Choose ONE reference mask for this camera/hour ---
        ref_mask, chosen_label, info = select_reference_mask(
            label_dir=label_dir,
            img_shape=sample_img.shape,
            max_labels=200,
            alpha_area=0.5,
            beta_iou=0.5,
        )

        # Save reference mask + info so you always know what was chosen
        info["camera"] = camera
        info["hour"] = hour
        info["chosen_label"] = chosen_label
        info["sample_image_used_for_shape"] = os.path.basename(sample_img_path)

        mask_path, json_path, overlay_path, outline_path = save_reference_outputs(
            out_dir=reference_out_dir,
            camera=camera,
            hour=hour,
            mask=ref_mask,
            info=info,
            sample_image_bgr=sample_img,
            save_debug=True,
        )

        print(f"✅ Reference mask selected for {camera}/{hour}")
        print(f"   chosen label: {chosen_label}")
        print(f"   mask saved:   {mask_path}")
        print(f"   info saved:   {json_path}")
        if overlay_path:
            print(f"   overlay:      {overlay_path}")
        if outline_path:
            print(f"   outline:      {outline_path}")

        # --- Apply the chosen mask to ALL images for this camera/hour ---
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            ts = extract_datetime(img_name)
            if ts is None:
                continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Safety: if shapes differ (shouldn't for static cams), resize mask to match
            if img.shape[:2] != ref_mask.shape[:2]:
                ref_mask_use = cv2.resize(ref_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                ref_mask_use = ref_mask

            VEG_METHOD = "exg"   # "hsv" | "exg" | "hybrid"
            # foliage = foliage_metric(img, ref_mask_use, method=VEG_METHOD)
            foliage_hsv = foliage_metric(img, ref_mask_use, method="hsv")
            foliage_exg = foliage_metric(img, ref_mask_use, method="exg")
            records.append({
                "camera": camera,
                "hour": hour,
                "datetime": ts,
                # vegetation metrics
                "green_foliage_hsv": foliage_hsv,
                "green_foliage_exg": foliage_exg,
                # provenance
                "reference_label": chosen_label,
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("❌ No data extracted — check paths and filenames.")

    # Baseline canopy (top 15% => 0.85 quantile by default)
    baselines = (
        df.groupby(["camera", "hour"])["green_foliage_exg"]
        .apply(lambda x: x.quantile(baseline_quantile))
    )

    df["baseline"] = df.set_index(["camera", "hour"]).index.map(baselines)
    df["canopy_fraction"] = df["green_foliage_exg"] / df["baseline"]

    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")

    return df


if __name__ == "__main__":
    IMAGE_BASE = r"C:\USERS\KIRA\PYTHON\TRAFFIC_SCRAPER\SORTED_BY_CAMERA"
    LABEL_BASE = r"C:\USERS\KIRA\PYTHON\TREE_IDENTIFIER_MODEL\RUNS\SEGMENT\ROAD_PREDICT1"
    OUTPUT_CSV = "foliage_timeseries.csv"

    build_timeseries_with_reference_masks(
        image_base=IMAGE_BASE,
        label_base=LABEL_BASE,
        output_csv=OUTPUT_CSV,
        baseline_quantile=0.85,
        reference_out_dir="reference_masks",
    )

    print("🌿 Foliage loss analysis complete.")
