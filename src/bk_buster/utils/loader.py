import os
from pathlib import Path

import pandas as pd


def load_data(data_path: str, delimiter: str|None=None) -> pd.DataFrame:

    # check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    # check if data_path is a string or path like object
    if isinstance(data_path, str):
        suffix = data_path.split('.')[-1]
    elif isinstance(data_path, Path):
        suffix = data_path.suffix
    elif isinstance(data_path, os.PathLike):
        suffix = Path(data_path).suffix
    else:
        raise TypeError("data_path must be a string or os.PathLike object")

    # load data based on file extension
    if suffix == '.csv':
        data = pd.read_csv(data_path, delimiter=delimiter)
    elif suffix == '.gz':
        data = pd.read_csv(data_path, compression='gzip', delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return data