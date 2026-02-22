import os
import glob
import pandas as pd
import yaml

FOLDER_PATTERN = "train*"
OUTPUT_FILE = "training_comparison.csv"


def get_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


# Extract results from training folder
def extract_results(folder):

    results_csv = os.path.join(folder, "results.csv")
    args_yaml = os.path.join(folder, "args.yaml")
    folder_name = os.path.basename(folder)

    result = {
        "Training": folder_name,
        "Model": None,

        #"Train_Images": None,
        #"Val_Images": None,
        #"Total_Images": None,
        "Epochs": None,
        "Batch": None,
        "LR0": None,
        "Cos_LR": None,
        "Best_Epoch": None,
        "Precision": None,
        "Recall": None,
        "mAP@0.5": None,
        "mAP@0.5:0.95": None,
    }

    # Read args.yaml
    if os.path.exists(args_yaml):
        try:
            with open(args_yaml, "r") as f:
                args = yaml.safe_load(f)

            model_name = os.path.basename(str(args.get("model", "")))

            result.update({
                "Model": model_name,
                "Epochs": args.get("epochs"),
                "Batch": args.get("batch"),
                "LR0": args.get("lr0"),
                "Cos_LR": args.get("cos_lr"),  # ✅ ADDED
            })

        except Exception as e:
            print(f"args.yaml error in {folder_name}: {e}")

    # Read results.csv
    if os.path.exists(results_csv):
        try:
            df = pd.read_csv(results_csv)

            if not df.empty:

                map50_col = get_column(df, [
                    "metrics/mAP50(B)",
                    "metrics/mAP50",
                    "metrics/mAP_0.5"
                ])

                map5095_col = get_column(df, [
                    "metrics/mAP50-95(B)",
                    "metrics/mAP50-95",
                    "metrics/mAP_0.5:0.95"
                ])

                precision_col = get_column(df, [
                    "metrics/precision(B)",
                    "metrics/precision"
                ])

                recall_col = get_column(df, [
                    "metrics/recall(B)",
                    "metrics/recall"
                ])

                if map50_col:
                    best_idx = df[map50_col].idxmax()
                    best_row = df.loc[best_idx]

                    result.update({
                        "Best_Epoch": int(best_row.get("epoch", best_idx)),
                        "Precision": round(best_row.get(precision_col, 0), 4) if precision_col else None,
                        "Recall": round(best_row.get(recall_col, 0), 4) if recall_col else None,
                        "mAP@0.5": round(best_row.get(map50_col, 0), 4),
                        "mAP@0.5:0.95": round(best_row.get(map5095_col, 0), 4) if map5095_col else None,
                    })

        except Exception as e:
            print(f"results.csv error in {folder_name}: {e}")

    return result


def main():

    folders = sorted(glob.glob(FOLDER_PATTERN))

    all_results = []

    for folder in folders:
        if os.path.isdir(folder):
            all_results.append(extract_results(folder))

    df = pd.DataFrame(all_results)

    df = df.sort_values(by="mAP@0.5:0.95", ascending=True, na_position="last")

    df.to_csv(OUTPUT_FILE, index=False)

    print("Results saved to:", OUTPUT_FILE)
    print(f"Total folders processed: {len(folders)}")
    print(f"Total rows in table: {len(df)}")
    print(df)


if __name__ == "__main__":
    main()