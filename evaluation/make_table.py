import json
import re
import string
import pandas as pd
import argparse
from pathlib import Path

def make_table(summary_df):
    #We first retrieve the model name from the file name
    summary_df["model"] = summary_df["filename"].apply(lambda x: x.split("result_")[1].split("_")[0])
    #Then we drop the columns of string
    summary_df.drop(columns=["filename"], inplace=True) 
    #Then we group by model and compute (max-min) for each column
    agg_df=summary_df.groupby("model").agg(lambda x: x.max() - x.min())
    return agg_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="./result")
    args = parser.parse_args()
    file_dir = Path(args.result_dir)

    summary_df=pd.read_csv(file_dir / "summary" / "sum.csv")
    agg_df=make_table(summary_df)
    agg_df.to_csv(file_dir / "summary" / "table3.csv")

