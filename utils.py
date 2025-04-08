import os
import logging
from glob import glob
from typing import List
from datetime import date
from typing import List, Dict, Optional

def find_files(directory: str, speaker: Optional[object] = None, ext: str = "wav") -> List[str]:
    """Find all files with given extension in directory recursively.
    
    Args:
        directory (str): Directory to search in
        ext (str): File extension to search for
        
    Returns:
        List[str]: List of found file paths
    """
    if speaker is not None:
        files = sorted(glob(directory + f"/**/{speaker}*.{ext}", recursive=True))
    else:
        files = sorted(glob(directory + f"/**/*.{ext}", recursive=True))
    logging.debug(f'find_files: {files}')
    return files

def ensures_dir(directory: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path to create
    """
    if len(directory) > 0 and not os.path.exists(directory):
        logging.debug(f'Making directory {directory}')
        os.makedirs(directory)

def setup_logging(log_file: str = "") -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_file (str): Path to log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_file == "":
        file_path = os.path.dirname(os.path.abspath(__file__))
        today = date.today()
        log_file = f'{file_path}/../logs/{today.strftime("%d-%m-%Y")}.log'
    
    logger = logging.getLogger("SpeakerVerificationLogger")
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when="midnight")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 