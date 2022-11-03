"""Miscellaneous utility functions. Anything hard to category."""
from pathlib import Path
import os
import logging


def create_parent_dir(path: str, file_ext: str = '.np') -> str:
    """Check file extension and parent directory. If it's not exist, create one."""
    path = Path(path).resolve()
    filename, _file_extension = os.path.splitext(path)
    if _file_extension != file_ext:
        path = Path(filename + file_ext)
        logging.warning('Change output path to: %s', path)
    path_output_dir = path.parent
    if not os.path.exists(path_output_dir):
        logging.info('Output directory is not found. Create: %s', path_output_dir)
        os.makedirs(path_output_dir)
    return path
