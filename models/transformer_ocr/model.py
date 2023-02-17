import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchvision.models import resnet101, resnet50
import math
from torch.autograd import Variable
from tokenizer import TransformerTokenizer


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerOCR(LightningModule):
    def __init__(self, model_config):
        super().__init__()
        hidden_dim = model_config.hidden_dim
        nheads = model_config.nheads
        num_encoder_layers = model_config.num_encoder_layers
        num_decoder_layers = model_config.num_decoder_layers
        vocab_len = model_config.vocab_len

        # create ResNet-101 backbone
        self.backbone = resnet101()
        del self.backbone.fc
        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads with length of vocab
        self.vocab = nn.Linear(hidden_dim, vocab_len)

        # output positional encodings (object queries)
        self.input_embed = nn.Embedding(vocab_len, hidden_dim)
        self.query_pos = PositionalEncoding(hidden_dim, 0.2)

        # spatial positional encodings, sine positional encoding can be used.
        # Detr baseline uses sine positional encoding.
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.trg_mask = None
        self.tokenizer = TransformerTokenizer()
        self.loss_func = LabelSmoothing(size=self.tokenizer.vocab_size, padding_idx=0, smoothing=0.1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"params": [p for p in self.parameters()], "name": "learning_rate"}],
            lr=1e-4,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[(lambda epoch: 0.95**epoch)])
        lr_scheduler = {"interval": "epoch", "scheduler": scheduler, "name": "AdamW"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_feature(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, inputs, trg):
        # propagate inputs through ResNet-101 up to avg-pool layer
        x = self.get_feature(inputs)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        bs, _, H, W = h.shape
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        # generating subsequent mask for target
        if (self.trg_mask is None or self.trg_mask.size(0) != len(trg)) or (
            self.trg_mask is not None and self.trg_mask.size(0) == trg.shape[0]
        ):
            self.trg_mask = self.transformer.generate_square_subsequent_mask(trg.shape[1], device=self.device).half()

        # Padding mask
        trg_pad_mask = self.make_len_mask(trg)

        # Getting postional encoding for target
        trg = self.input_embed(trg)
        trg = self.query_pos(trg)

        output = self.transformer(
            pos + 0.1 * h.flatten(2).permute(2, 0, 1),
            trg.permute(1, 0, 2),
            tgt_mask=self.trg_mask,
            tgt_key_padding_mask=trg_pad_mask.permute(1, 0),
        )

        return self.vocab(output.transpose(0, 1))

    def training_step(self, batch, batch_idx):
        images, labels_y, label_len = batch
        output = self(images, labels_y[:, :-1])
        norm = (labels_y != 0).sum()
        loss = (
            self.loss_func(
                output.log_softmax(-1).contiguous().view(-1, self.tokenizer.vocab_size),
                labels_y[:, 1:].contiguous().view(-1).long(),
            )
            / norm
        )
        self.log("train_loss", loss, sync_dist=(self.device != "cpu"))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels_y, label_len = batch
        logits = self(images, labels_y[:, :-1])
        return {
            "logits": logits,
            "labels": labels_y
        }

    def validation_epoch_end(self, validation_step_outputs):
        result_loss = 0.0
        result_acc = 0.0
        for out in validation_step_outputs:
            labels_ids = out["labels"]
            norm = (out["labels"] != 0).sum()
            loss = (
                self.loss_func(
                    out["logits"].log_softmax(-1).contiguous().view(-1, self.tokenizer.vocab_size),
                    out["labels"][:, 1:].contiguous().view(-1).long(),
                )
                / norm
            )
            result_loss += loss
            log_prob_logits = torch.log_softmax(out["logits"], dim=2, dtype=torch.float32)
            pred_ids = torch.argmax(log_prob_logits, dim=-1)
            references = self.tokenizer.batch_decode(labels_ids)
            predictions = self.tokenizer.batch_decode(pred_ids)
            acc = 0
            for i in range(len(references)):
                if references[i] == predictions[i]:
                    acc += 1
            acc = acc / len(references)
            result_acc += acc
        self.log("val_loss", result_loss / len(validation_step_outputs), sync_dist=(self.device != "cpu"))
        self.log("accuracy", result_acc / len(validation_step_outputs), sync_dist=(self.device != "cpu"))

    def get_memory(self, imgs):
        x = self.conv(self.get_feature(imgs))
        bs, _, H, W = x.shape
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        return self.transformer.encoder(pos + 0.1 * x.flatten(2).permute(2, 0, 1))

    def predict_step(self, batch, batch_idx):
        imgs, _, _ = batch
        with torch.no_grad():
            memory = self.get_memory(imgs)
            out_indexes = [self.tokenizer.chars.index(self.tokenizer.SOS)]

            for i in range(128):
                mask = self.transformer.generate_square_subsequent_mask(i + 1).to(self.device)
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(self.device)
                output = self.vocab(
                    self.transformer.decoder(self.query_pos(self.decoder(trg_tensor)), memory, tgt_mask=mask)
                )
                out_token = output.argmax(2)[-1].item()
                if out_token == self.tokenizer.chars.index(self.tokenizer.EOS):
                    break

                out_indexes.append(out_token)

        return {"out_indexes": out_indexes}

    def predict_epoch_end(self, predict_step_outputs):
        pre = self.tokenizer.decode(predict_step_outputs["out_indexes"][1:])
        print(pre)
