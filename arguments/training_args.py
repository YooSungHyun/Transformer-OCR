from dataclasses import dataclass


@dataclass
class TrainingArguments:
    """Help string for this group of command-line arguments"""

    seed: int = None  # all seed
    local_rank: int = None  # ddp local rank
    train_data_path: str = "preprocess/train.csv"  # target pytorch lightning data dirs
    valid_data_path: str = "preprocess/valid.csv"  # target pytorch lightning data dirs
    test_data_path: str = "preprocess/test.csv"
    output_dir: str = "model_outputs"  # model output path
    config_path: str = "config/model_config.json"
    num_workers: int = None  # how many proc map?
    per_device_train_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for training.
    per_device_eval_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for evaluation.
    per_device_test_batch_size: int = 1
    valid_on_cpu: bool = False  # If you want to run validation_step on cpu -> true
    image_h: int = 192
    image_w: int = 384
    use_pretrained_backbone: bool = True
    deepspeed_config: str = "ds_config/zero2.json"
