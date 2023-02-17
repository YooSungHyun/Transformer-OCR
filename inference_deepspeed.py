import os
import torch
import pytorch_lightning as pl
from utils.compy import dataclass_to_namespace
from utils.config_loader import load_config
from arguments.inference_args import InferenceArguments
from simple_parsing import ArgumentParser
from models.transformer_ocr.model import TransformerOCR
from models.transformer_ocr.datamodule import TransOCRDataModule
from pytorch_lightning.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        result_list = list()
        for batch_index, pred in list(zip(batch_indices[0], predictions[0])):
            result_list.append(list(zip(batch_index, pred)))
        torch.save(result_list, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))


def on_load_checkpoint(checkpoint):
    state_dict = {k.partition("_forward_module.")[2]: checkpoint[k] for k in checkpoint.keys()}
    checkpoint["state_dict"] = state_dict
    return checkpoint


def main(hparams):
    pl.seed_everything(hparams.seed)
    device = torch.device("cuda")
    os.makedirs("distributed_result", exist_ok=True)
    model_config_dict = load_config(hparams.model_config)
    datamodule = TransOCRDataModule(hparams)
    model = TransformerOCR(model_config_dict)
    checkpoint = torch.load(hparams.model_path, map_location=device)
    checkpoint = on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    pred_writer = CustomWriter(output_dir="distributed_result", write_interval="epoch")
    hparams.callbacks = [pred_writer]
    trainer = pl.Trainer.from_argparse_args(hparams)

    trainer.predict(model, datamodule=datamodule, return_predictions=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(InferenceArguments, dest="inference_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "inference_args")
    main(args)
