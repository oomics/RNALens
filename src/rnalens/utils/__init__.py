from glob import glob
import re
import json

def select_best_checkpoint(checkpoint_dir: str) -> str:
    """Selects the best model checkpoint based on trainer state information.

    Scans the specified directory for checkpoint folders, identifies the most recent
    checkpoint by epoch number, then returns the path to the best-performing checkpoint
    according to the trainer's recorded best_global_step.

    Args:
        checkpoint_dir: Path to directory containing checkpoint folders (named 'checkpoint-*')

    Returns:
        str: Path to the best-performing checkpoint directory

    Raises:
        ValueError: If no checkpoint directories are found in the specified path
        FileNotFoundError: If trainer_state.json is missing in the checkpoint directory
        KeyError: If best_global_step is missing from trainer_state.json

    Example:
        >>> best_path = select_best_checkpoint("./output/checkpoints")
        >>> print(f"Best checkpoint: {best_path}")
    """
    targets = glob(f"{checkpoint_dir}/checkpoint-*")
    if len(targets) == 0:
        raise ValueError(f"cannot find checkpoint-* in {checkpoint_dir}")

    def extract_epoch_num(checkpoint_path):
        match = re.search(r'checkpoint-(\d+)', checkpoint_path)
        return int(match.group(1)) if match else -1
    # Sort checkpoints by epoch number in descending order
    sorted_checkpoints = sorted(targets, key=extract_epoch_num, reverse=True)
    
    last_checkpoint = sorted_checkpoints[0]
    with open(f"{last_checkpoint}/trainer_state.json") as f:
        checkpoint_info = json.load(f)
        best_global_step = checkpoint_info["best_global_step"]

    return f"{checkpoint_dir}/checkpoint-{best_global_step}"