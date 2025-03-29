from pathlib import Path
import numpy as np
#import fast  
import pandas as pd

def main():
    root = Path.cwd()
    present_dir = root / "present"
    output_dir = root / "npy_arrays"
    output_dir.mkdir(parents=True, exist_ok=True)

    features_list = []  

    for patient_folder in present_dir.iterdir():
        if not patient_folder.is_dir():
            continue

        patient_id = patient_folder.name

        for series_folder in patient_folder.iterdir():
            if not series_folder.is_dir():
                continue

            series_name = series_folder.name
            safe_series_name = series_name.replace(" ", "_")

            try:
                ct_image = fast.readImage(str(series_folder))
            except Exception as e:
                print(f"Skipping {series_folder} - could not load: {e}")
                continue

            ct_array = ct_image.toArray()

            filename = f"{patient_id}_{safe_series_name}.npy"
            out_path = output_dir / filename
            np.save(out_path, ct_array)
            print(f"Saved {out_path}")

            features = {
                "PatientID": patient_id,
                "Series": series_name,
                "Filename": filename,
                "Mean": float(np.mean(ct_array)),
                "Std": float(np.std(ct_array)),
                "Min": float(np.min(ct_array)),
                "Max": float(np.max(ct_array))
            }
            features_list.append(features)

    df = pd.DataFrame(features_list)
    csv_path = root / "fast_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved FAST features to {csv_path}")

if __name__ == "__main__":
    main()
