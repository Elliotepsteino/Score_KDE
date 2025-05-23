import numpy as np
import pandas as pd
import os
from glob import glob


def process_arem():
    """
    Process AReM dataset.
    Files have metadata headers starting with # and CSV data below.
    """
    print("Processing AReM dataset...")

    # Create output directory
    os.makedirs("/scratch/Score_KDE/minDiffusion/uci/datasets/AReM", exist_ok=True)

    # Define activities
    activities = [
        "bending1",
        "bending2",
        "cycling",
        "lying",
        "sitting",
        "standing",
        "walking",
    ]

    all_data = []
    column_names = None

    for activity in activities:
        # Get all CSV files in the activity folder
        csv_files = glob(
            f"/scratch/Score_KDE/minDiffusion/uci/arem/{activity}/dataset*.csv"
        )

        for file in csv_files:
            try:
                # Read the file first to extract metadata and find the data start
                with open(file, "r") as f:
                    lines = f.readlines()

                # Extract column names from metadata
                for line in lines:
                    if "# Columns:" in line:
                        column_names = line.replace("# Columns:", "").strip().split(",")
                        break

                # Find where the actual data starts (first line without #)
                data_start = 0
                for i, line in enumerate(lines):
                    if not line.startswith("#"):
                        data_start = i
                        break

                # Read the CSV data starting after the metadata
                data = pd.read_csv(
                    file, skiprows=data_start, header=None, names=column_names
                )
                all_data.append(data.values)

            except Exception as e:
                print(f"Error processing file {file}: {e}")

    # Combine all data
    if all_data:
        combined_data = np.vstack(all_data)

        # Save processed data
        np.save("datasets/AReM/data.npy", combined_data)
        print(f"AReM dataset processed. Shape: {combined_data.shape}")

        # Save column names
        if column_names:
            with open("datasets/AReM/feature_names.txt", "w") as f:
                f.write("\n".join(column_names))

        # Print first few rows to verify
        print("\nFirst few rows of processed data:")
        print(pd.DataFrame(combined_data[:5], columns=column_names))

    else:
        print("No data was processed for AReM dataset!")


def process_casp():
    """
    Process CASP (Protein Structure Prediction) dataset.
    """
    print("Processing CASP dataset...")

    # Create output directory
    os.makedirs("/scratch/Score_KDE/minDiffusion/uci/datasets/CASP", exist_ok=True)

    try:
        # Read the data file
        data = pd.read_csv("/scratch/Score_KDE/minDiffusion/uci/casp/CASP.csv")

        # Remove any target/label column if present (usually the last column)
        features = data.iloc[:, :-1].values

        # Save processed data
        np.save("/scratch/Score_KDE/minDiffusion/uci/datasets/CASP/data.npy", features)
        print(f"CASP dataset processed. Shape: {features.shape}")

        # Save feature names
        feature_names = data.columns[:-1].tolist()
        with open(
            "/scratch/Score_KDE/minDiffusion/uci/datasets/CASP/feature_names.txt", "w"
        ) as f:
            f.write("\n".join(feature_names))

    except Exception as e:
        print(f"Error processing CASP dataset: {e}")


def process_bank():
    """
    Process Bank Marketing dataset.
    """
    print("Processing Bank dataset...")

    # Create output directory
    os.makedirs("/scratch/Score_KDE/minDiffusion/uci/datasets/BANK", exist_ok=True)

    try:
        # Read the data file
        data = pd.read_csv("/scratch/Score_KDE/minDiffusion/uci/bank/bank.csv", sep=";")

        # Separate numerical and categorical columns
        num_cols = data.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = data.select_dtypes(include=["object"]).columns

        # Drop the target variable 'y' if present
        if "y" in data.columns:
            data = data.drop("y", axis=1)
            if "y" in num_cols:
                num_cols = num_cols.drop("y")
            if "y" in cat_cols:
                cat_cols = cat_cols.drop("y")

        # Process numerical features
        numerical_data = data[num_cols]

        # One-hot encode categorical features
        categorical_data = pd.get_dummies(data[cat_cols])

        # Combine numerical and categorical features
        processed_data = pd.concat([numerical_data, categorical_data], axis=1)

        # Convert to numpy array
        features = processed_data.values

        # Save processed data
        np.save("/scratch/Score_KDE/minDiffusion/uci/datasets/BANK/data.npy", features)
        print(f"Bank dataset processed. Shape: {features.shape}")

        # Save feature names
        feature_names = processed_data.columns.tolist()
        with open(
            "/scratch/Score_KDE/minDiffusion/uci/datasets/BANK/feature_names.txt", "w"
        ) as f:
            f.write("\n".join(feature_names))

    except Exception as e:
        print(f"Error processing Bank dataset: {e}")


def main():
    """
    Process all datasets and create the required directory structure.
    """
    # Create main datasets directory
    os.makedirs("/scratch/Score_KDE/minDiffusion/uci/datasets", exist_ok=True)

    # Process each dataset
    process_arem()
    process_casp()
    # process_bank()

    print("\nAll datasets processed successfully!")


if __name__ == "__main__":
    main()
