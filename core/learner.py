import copy
import gc
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

import wandb

from core import calculate_metrics
from core import logger
from core.model import LABEL2EMO


class Learner:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dataloaders,
        model_name,
        model,
        batch_size,
        cuda_device="cuda:0",
    ):
        self.device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)

        self.__model_name = model_name

        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        logger.info(f"train labels{np.unique(self.train_dataset.df.label.values, return_counts=True)}")

        self.dataloaders = dataloaders

    def train(self, num_epochs, learning_rate, optimizer_step, optimizer_gamma, weight_decay=0, clip_grad=False):
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=optimizer_step, gamma=optimizer_gamma)

        since = time.time()
        best_model_wts = None
        best_loss = 10000000
        best_acc = best_f1 = best_WA = 0
        softmax = nn.Softmax(dim=1)
        torch.cuda.empty_cache()
        gc.collect()
        iters = {"train": 1, "val": 1}
        try:
            for epoch in tqdm(range(1, num_epochs + 1)):
                for phase in ["train", "val"]:
                    if phase == "train":
                        self.model.train()
                        cur_step_lr = scheduler.get_last_lr()[-1]
                        wandb.log(
                            {
                                "EPOCH/learning_rate": cur_step_lr,
                                "epoch": epoch,
                            }
                        )
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_outputs = []
                    running_labels = []
                    for inputs, labels in tqdm(self.dataloaders[phase], leave=False):
                        inputs = inputs.to(self.device)
                        labels = labels.long()
                        labels = labels.to(self.device)
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self.model(inputs)
                            probs = softmax(outputs)
                            loss = criterion(outputs, labels)
                            if phase == "train":
                                loss.backward()
                                if clip_grad:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                                optimizer.step()
                            wandb.log({f"BATCH/{phase}_loss": loss.item(), f"BATCH/{phase}_iter": iters[phase]})

                        running_loss += loss.detach().item()
                        if phase == "val":
                            running_labels.append(labels)
                            running_outputs.append(probs)
                        iters[phase] += 1
                        torch.cuda.empty_cache()
                        gc.collect()

                    if phase == "train":
                        scheduler.step()

                    epoch_loss = running_loss / len(self.dataloaders[phase])
                    wandb.log(
                        {
                            f"EPOCH/{phase}_loss": epoch_loss,
                            "epoch": epoch,
                        }
                    )

                    if phase == "val":
                        pred_class = np.argmax(torch.cat(running_outputs).cpu().numpy(), axis=1)
                        gt_class = torch.cat(running_labels).cpu().numpy()

                        metric_dict = calculate_metrics(pred_class, gt_class, neg_label=0)

                        epoch_acc = metric_dict["accuracy"]
                        epoch_f1 = metric_dict["f1_macro"]
                        epoch_WA = metric_dict["WA"]
                        wandb.log(
                            {
                                "EPOCH/accuracy": epoch_acc,
                                "EPOCH/F1": epoch_f1,
                                "EPOCH/W_Accuracy": epoch_WA,
                                "epoch": epoch,
                            }
                        )
                        for i in range(len(LABEL2EMO.keys())):
                            wandb.log(
                                {
                                    f"RECALL/{LABEL2EMO[i]}": metric_dict["recall_by_class"][i],
                                    f"F1_SCORE/{LABEL2EMO[i]}": metric_dict["f1_by_class"][i],
                                    f"PRECISION/{LABEL2EMO[i]}": metric_dict["precision_by_class"][i],
                                }
                            )

                        logger.info(f"{phase} Loss: {epoch_loss:.4f}")
                        logger.info(f"{phase} Acc: {epoch_acc:.4f}")
                        logger.info(f"{phase} F1 macro: {epoch_f1:.4f}")
                        logger.info(f"{phase} WA: {epoch_WA:.4f}")
                        if epoch_f1 > best_f1:
                            best_f1 = epoch_f1
                            best_WA = epoch_WA
                            best_acc = epoch_acc
                            best_f1 = epoch_f1
                            best_loss = epoch_loss
                            best_epoch = epoch
                            best_model_wts = copy.deepcopy(self.model.state_dict())

                    else:
                        logger.info(f"{phase} Loss: {epoch_loss:.4f}")

        except KeyboardInterrupt:
            pass

        # summary_writer.flush()
        time_elapsed = time.time() - since
        logger.success(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s."
            + f" Best model loss: {best_loss:.6f}, best model acc: {best_acc:.6f}, "
            + f"best model f1: {best_f1:.6f},  best model WA {best_WA:.6f} best epoch {best_epoch}"
        )

        self.model.load_state_dict(best_model_wts)
        self.model.eval()
        return best_model_wts
