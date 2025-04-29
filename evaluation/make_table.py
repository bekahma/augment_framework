import json
import re
import string
import pandas as pd
import argparse
from pathlib import Path

def make_tables(summary_df):
    # Extract model name from filename
    summary_df["model"] = summary_df["filename"].apply(lambda x: x.split("/")[1])
    summary_df.drop(columns=["filename"], inplace=True) 

    # Table 3: Compute (max - min) per column
    range_df = summary_df.groupby("model").agg(lambda x: x.max() - x.min())

    # Table 13: Compute max and min separately for each column
    max_df = summary_df.groupby("model").max().add_suffix("_max")
    min_df = summary_df.groupby("model").min().add_suffix("_min")
    cols = [col.replace("_max", "") for col in max_df.columns]
    interleaved_cols = [col + suffix for col in cols for suffix in ["_max", "_min"]]  
    min_max_df = pd.concat([max_df, min_df], axis=1)[interleaved_cols] 

    return range_df, min_max_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="./result")
    args = parser.parse_args()
    file_dir = Path(args.result_dir)

    summary_df=pd.read_csv(file_dir / "summary" / "paraphrase_sum.csv")
    range_df, min_max_df = make_tables(summary_df)

    # Save both tables
    range_df.to_csv(file_dir / "summary" / "table3.csv")
    min_max_df.to_csv(file_dir / "summary" / "table13.csv")

