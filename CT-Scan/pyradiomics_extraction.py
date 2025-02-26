from pathlib import Path
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd

def main():
    root = Path.cwd()
    array_dir = root / "npy_arrays"
    extractor = featureextractor.RadiomicsFeatureExtractor()

    features_list = []  # To store feature dictionaries

    # Process each .npy file in the npy_arrays folder
    for npy_path in array_dir.glob("*.npy"):
        ct_array = np.load(npy_path)
        ct_image = sitk.GetImageFromArray(ct_array)
        ct_image.SetSpacing((1.0, 1.0, 1.0))

        # Create a dummy mask (all ones)
        mask_array = np.ones_like(ct_array, dtype=np.uint8)
        mask_image = sitk.GetImageFromArray(mask_array)
        mask_image.SetSpacing((1.0, 1.0, 1.0))

        radiomics_features = extractor.execute(ct_image, mask_image)
        
        # Parse patient and series from the filename (format: PatientID_Series.npy)
        parts = npy_path.stem.split('_', 1)
        if len(parts) == 2:
            patient_id, series = parts
        else:
            patient_id = parts[0]
            series = ""

        feature_dict = {"PatientID": patient_id, "Series": series, "Filename": npy_path.name}
        # Merge the radiomics features into our dict
        for key, value in radiomics_features.items():
            feature_dict[key] = value
        features_list.append(feature_dict)
        
        print(f"Processed features for {npy_path.name}")

    # Write features to CSV
    df = pd.DataFrame(features_list)
    csv_path = root / "pyradiomics_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved PyRadiomics features to {csv_path}")

if __name__ == "__main__":
    main()
