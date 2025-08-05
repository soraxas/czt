import argparse
import os

import pandas as pd

files = ["train", "test"]

bic_to_bank = {
    "ZHSZUS33": "Bank_1",
    "SHSHKHH1": "Bank_2",
    "YXRXGB22": "Bank_3",
    "WPUWDEFF": "Bank_4",
    "YMNYFRPP": "Bank_5",
    "FBSFCHZH": "Bank_6",
    "YSYCESMM": "Bank_7",
    "ZNZZAU3M": "Bank_8",
    "HCBHSGSG": "Bank_9",
    "XITXUS33": "Bank_10",
}

original_columns = [
    "Transaction_ID",
    "Timestamp",
    "Transaction_Amount",
    "trans_volume",
    "total_amount",
    "average_amount",
    "hist_trans_volume",
    "hist_total_amount",
    "hist_average_amount",
    "x2_y1",
    "x3_y2",
]


def main():
    args = define_parser()
    root_path = args.input_dir
    original_feat_postfix = "_normalized.csv"
    embed_feat_postfix = "_embedding.csv"
    out_feat_postfix = "_combined.csv"

    # Loop through all folders in root_path that match the pattern xxx_Bank_n
    for folder_name in os.listdir(root_path):
        if not os.path.isdir(os.path.join(root_path, folder_name)):
            continue
        # Check if folder name matches the pattern *_Bank_n
        if "_Bank_" in folder_name:
            print("Processing folder: ", folder_name)
            # Extract BIC from folder name (assume BIC is before the first underscore)
            bic = folder_name.split("_")[0]
        else:
            continue
        for file in files:
            original_feat_file = os.path.join(
                root_path, bic + "_" + bic_to_bank[bic], file + original_feat_postfix
            )
            embed_feat_file = os.path.join(
                root_path, bic + "_" + bic_to_bank[bic], file + embed_feat_postfix
            )
            out_feat_file = os.path.join(
                root_path, bic + "_" + bic_to_bank[bic], file + out_feat_postfix
            )

            # Load the original and embedding features
            original_feat = pd.read_csv(original_feat_file)
            embed_feat = pd.read_csv(embed_feat_file)

            # Select the columns of the original features
            original_feat = original_feat[original_columns]

            # Combine the features, matching the rows by "Transaction_ID"
            out_feat = pd.merge(original_feat, embed_feat, on="Transaction_ID")

            # Save the combined features
            out_feat.to_csv(out_feat_file, index=False)


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="output directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
