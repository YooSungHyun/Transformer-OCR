from dataclasses import dataclass


@dataclass
class InferenceArguments:
    """Help string for this group of command-line arguments"""

    seed: int = None  # all seed
    local_rank: int = None  # ddp local rank
    model_path: str = "model_outputs"  # target pytorch lightning model dir
    config_path: str = "model_outputs"  # target pytorch lightning model dir
    train_data_path: str = None
    valid_data_path: str = None
    test_data_path: str = "preprocess/test.csv"
    num_workers: int = 4  # how many proc map?
    per_device_train_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for training.
    per_device_eval_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for evaluation.
    per_device_test_batch_size: int = 1
    image_h: int = 192
    image_w: int = 384
    valid_on_cpu: bool = False  # If you want to run validation_step on cpu -> true
