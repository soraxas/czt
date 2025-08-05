import os
from typing import Tuple

import pandas as pd
import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader


class CreditCardEmbedDataLoader(XGBDataLoader):
    def __init__(self, root_dir: str, file_postfix: str):
        self.dataset_names = ["train", "test"]
        self.base_file_names = {}
        self.root_dir = root_dir
        self.file_postfix = file_postfix
        for name in self.dataset_names:
            self.base_file_names[name] = name + file_postfix

        self.numerical_columns = [
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
        ] + [f"V_{i}" for i in range(64)]

    def initialize(
        self,
        client_id: str,
        rank: int,
        data_split_mode: xgb.core.DataSplitMode = xgb.core.DataSplitMode.ROW,
    ):
        super().initialize(client_id, rank, data_split_mode)

    def load_data(self) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        data = {}
        for ds_name in self.dataset_names:
            print("\nloading for site = ", self.client_id, f"{ds_name} dataset \n")
            file_name = os.path.join(
                self.root_dir, self.client_id, self.base_file_names[ds_name]
            )
            print(file_name)
            df = pd.read_csv(file_name)
            data_num = len(data)

            # split to feature and label
            y = df["Fraud_Label"]
            x = df[self.numerical_columns]
            data[ds_name] = (x, y, data_num)

        # training
        x_train, y_train, total_train_data_num = data["train"]
        dmat_train = xgb.DMatrix(
            x_train, label=y_train, data_split_mode=self.data_split_mode
        )

        # validation
        x_valid, y_valid, total_valid_data_num = data["test"]
        dmat_valid = xgb.DMatrix(
            x_valid, label=y_valid, data_split_mode=self.data_split_mode
        )

        return dmat_train, dmat_valid
