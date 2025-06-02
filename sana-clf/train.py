import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from tqdm.auto import tqdm

import wandb
from transformers import get_linear_schedule_with_warmup
from enum import Enum

from .pipeline import SanaClassifierPipeline, ModelParameters


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"


class SchedulerType(Enum):
    NONE = "none"
    LINEAR_WARMUP = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_restarts"
    EXPONENTIAL = "exponential"


class ImageLabelDataset(Dataset):
    """
    Dataset для изображений с бинарными метками.
    Ожидает DataFrame с колонками ['fp', 'label'].
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, torch.Tensor, str]:
        row = self.df.iloc[idx]
        fp = row["fp"]
        label = float(row["label"])
        image = Image.open(fp).convert("RGB")
        return image, torch.tensor(label, dtype=torch.float32), fp


def collate_image_label(
    batch: List[Tuple[Image.Image, torch.Tensor, str]]
) -> Tuple[List[Image.Image], torch.Tensor, List[str]]:
    """
    Collate-функция для DataLoader: собирает список изображений, меток и путей.
    """
    images = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    paths = [item[2] for item in batch]
    return images, labels, paths


def set_seed(seed: int):
    """
    Фиксируем сид для воспроизводимости.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_model(
    pretrained_model: str,
    transformer_layers: List[int],
    model_parameters: ModelParameters,
    device: str = "cuda",
) -> SanaClassifierPipeline:
    """
    Загружает SanaClassifierPipeline с заданной базовой моделью,
    переводит в нужные dtype и регистрирует классификатор.
    """
    device_obj = torch.device(device)
    try:
        pipe = SanaClassifierPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float32)
    except Exception:
        from transformers import modeling_utils

        # Обход потенциальной проблемы с ALL_PARALLEL_STYLES
        if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
            modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
        pipe = SanaClassifierPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float32)
    pipe.to(device_obj)
    pipe.text_encoder = pipe.text_encoder.to(torch.bfloat16)
    pipe.transformer = pipe.transformer.to(torch.bfloat16)
    pipe.register_model(transformer_blocks_ids=transformer_layers, clf_params=model_parameters.clf_params)
    return pipe


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Создает DataLoader'ы для обучения и валидации.
    """
    train_dataset = ImageLabelDataset(train_df)
    val_dataset = ImageLabelDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_image_label,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_image_label,
        drop_last=False,
    )
    return train_loader, val_loader


def build_optimizer_scheduler(
    pipe: SanaClassifierPipeline,
    optimizer_type: OptimizerType,
    lr: float,
    weight_decay: float,
    scheduler_type: SchedulerType,
    warmup_steps: int,
    total_steps: int,
) -> Tuple[optim.Optimizer, Optional[Any]]:
    """
    Создает оптимизатор и LR-шейулер в зависимости от переданных параметров.
    """
    params_to_optimize = [p for p in pipe.parameters() if p.requires_grad]
    if optimizer_type == OptimizerType.ADAM:
        optimizer = optim.Adam(params_to_optimize, lr=lr)
    else:  # ADAMW
        optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)

    scheduler = None
    if scheduler_type == SchedulerType.LINEAR_WARMUP:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif scheduler_type == SchedulerType.COSINE:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=total_steps, T_mult=1)
    elif scheduler_type == SchedulerType.EXPONENTIAL:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    return optimizer, scheduler


def save_checkpoint(
    output_dir: str,
    step_or_epoch: Union[int, str],
    pipe: SanaClassifierPipeline,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    meta: Dict[str, Any],
    verbose: bool = True,
):
    """
    Сохраняет checkpoint:
    - Веса классификатора через pipe.save_clf()
    - Состояние оптимизатора/шейдьюлера и мета-информацию в optimizer.pt
    """
    ckpt_dir = Path(output_dir) / f"checkpoint-{step_or_epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    clf_path = ckpt_dir / "classifier.safetensors"
    pipe.save_clf(str(clf_path))

    opt_sd: Dict[str, Any] = {
        "optimizer_state": optimizer.state_dict(),
        **{"epoch": meta.get("epoch", None)},
        **{"global_step": meta.get("global_step", None)},
        "meta": meta,
    }
    if scheduler is not None:
        opt_sd["scheduler_state"] = scheduler.state_dict()

    torch.save(opt_sd, ckpt_dir / "optimizer.pt")
    if verbose:
        print(f"Checkpoint сохранен: {ckpt_dir}")


def train_one_epoch(
    pipe: SanaClassifierPipeline,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    scheduler_type: SchedulerType,
    criterion: nn.Module,
    model_parameters: ModelParameters,
    device_obj: torch.device,
    verbose: bool,
    progress_bar: bool,
    epoch: int,
    global_step: int,
    logging_steps: int,
    wandb_enabled: bool,
) -> Tuple[float, int]:
    """
    Один проход по эпохе обучения.
    Возвращает суммарный loss за эпоху и обновленный global_step.
    """
    pipe.train()
    epoch_loss = 0.0
    total_batches = len(train_loader)
    iterator = enumerate(train_loader)
    if progress_bar:
        iterator = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch} [train]", leave=False)

    for batch_idx, (images, labels, _) in iterator:
        batch_size_actual = len(images)
        labels = labels.to(device_obj).view(batch_size_actual, 1)

        optimizer.zero_grad()
        losses = []
        for i in range(batch_size_actual):
            img = images[i]
            label = labels[i : i + 1]
            logits = pipe(img, model_parameters.prompt, model_parameters.t)
            loss = criterion(logits, label)
            losses.append(loss)

        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        optimizer.step()

        if scheduler is not None:
            # для всех шейдеров выполняем шаг после optimizer.step()
            scheduler.step()

        epoch_loss += batch_loss.item()
        global_step += 1

        if wandb_enabled and global_step % logging_steps == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            wandb.log({"train/loss": avg_loss, "step": global_step})

        if verbose and not progress_bar and global_step % logging_steps == 0:
            print(f"[Epoch {epoch} | Step {global_step}] train loss: {batch_loss.item():.4f}")

    return epoch_loss, global_step


def validate_one_epoch(
    pipe: SanaClassifierPipeline,
    val_loader: DataLoader,
    model_parameters: ModelParameters,
    device_obj: torch.device,
    output_dir: str,
    epoch: int,
    verbose: bool,
    progress_bar: bool,
    wandb_enabled: bool,
) -> Dict[str, Any]:
    """
    Один проход по валидации:
    - Считает вероятности, предсказания и метрики.
    - Сохраняет CSV с результатами.
    Возвращает словарь с метриками.
    """
    pipe.eval()
    all_probs: List[float] = []
    all_preds: List[int] = []
    all_labels: List[float] = []
    all_paths: List[str] = []

    total_batches = len(val_loader)
    iterator = val_loader
    if progress_bar:
        iterator = tqdm(val_loader, total=total_batches, desc=f"Epoch {epoch} [val]", leave=False)

    with torch.no_grad():
        for images, labels, paths in iterator:
            batch_size_actual = len(images)
            probs_batch = []
            preds_batch = []
            for i in range(batch_size_actual):
                img = images[i]
                logit = pipe(img, model_parameters.prompt, model_parameters.t)
                prob = torch.sigmoid(logit).item()
                pred = 1 if prob >= 0.5 else 0
                probs_batch.append(prob)
                preds_batch.append(pred)

            all_probs.extend(probs_batch)
            all_preds.extend(preds_batch)
            all_labels.extend(labels.tolist())
            all_paths.extend(paths)

    # Сохраняем результаты в CSV
    df_val = pd.DataFrame({
        "fp": all_paths,
        "label": all_labels,
        "prob": all_probs,
        "pred": all_preds,
    })
    val_csv_path = Path(output_dir) / f"val_epoch_{epoch}.csv"
    os.makedirs(val_csv_path.parent, exist_ok=True)
    df_val.to_csv(val_csv_path, index=False)

    # Вычисляем метрики
    y_true = df_val["label"]
    y_pred = df_val["pred"]
    y_prob = df_val["prob"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    if wandb_enabled:
        wandb.log({
            "val/accuracy": acc,
            "val/precision": prec,
            "val/recall": rec,
            "val/f1": f1,
            "val/roc_auc": auc,
            "val/epoch": epoch,
        })

    if verbose:
        print(f"Validation results for epoch {epoch}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {auc:.4f}")
        print("  Confusion Matrix:")
        print(cm)
        print("  Classification Report:")
        print(report)

    return metrics


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pretrained_model: str,
    model_parameters: ModelParameters,
    transformer_layers: List[int],
    output_dir: str,
    batch_size: int = 8,
    epochs: int = 5,
    lr: float = 1e-4,
    optimizer_type: Union[str, OptimizerType] = OptimizerType.ADAMW,
    weight_decay: float = 0.0,
    scheduler_type: Union[str, SchedulerType] = SchedulerType.LINEAR_WARMUP,
    warmup_steps: int = 0,
    logging_steps: int = 50,
    save_steps: int = 200,
    seed: int = 42,
    device: str = "cuda",
    wandb_project: str = "sana-classifier",
    wandb_run_name: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    verbose: bool = True,
    progress_bar: bool = True,
):
    """
    Основная функция обучения:
    - Создает DataLoader'ы
    - Загружает модель
    - Настраивает оптимайзер и шейдер
    - Производит обучение с валидацией и сохранением чекпоинтов
    - Логгирует в wandb
    """
    set_seed(seed)

    # Проверка наличия необходимых колонок
    if "fp" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train_df должен содержать колонки 'fp' и 'label'")
    if "fp" not in val_df.columns or "label" not in val_df.columns:
        raise ValueError("val_df должен содержать колонки 'fp' и 'label'")

    # Создаем DataLoader'ы
    train_loader, val_loader = build_dataloaders(train_df, val_df, batch_size)

    # Готовим модель
    pipe = prepare_model(pretrained_model, transformer_layers, model_parameters, device)
    device_obj = torch.device(device)

    # Заморозим все параметры, кроме классификатора
    for name, param in pipe.named_parameters():
        if "clf_model" not in name:
            param.requires_grad = False

    # Приводим типы enum
    if isinstance(optimizer_type, str):
        optimizer_type = OptimizerType(optimizer_type.lower())
    if isinstance(scheduler_type, str):
        scheduler_type = SchedulerType(scheduler_type.lower())

    total_steps = len(train_loader) * epochs
    optimizer, scheduler = build_optimizer_scheduler(
        pipe, optimizer_type, lr, weight_decay, scheduler_type, warmup_steps, total_steps
    )

    # Настройки для продолжения обучения
    start_epoch = 1
    global_step = 0
    if resume_checkpoint is not None:
        ckpt_dir = Path(resume_checkpoint)
        pipe.load_clf(str(ckpt_dir / "classifier.safetensors"))
        ckpt = torch.load(ckpt_dir / "optimizer.pt", map_location=device_obj)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("meta", {}).get("epoch") is not None:
            start_epoch = ckpt["meta"]["epoch"] + 1
        if ckpt.get("meta", {}).get("global_step") is not None:
            global_step = ckpt["meta"]["global_step"]

    # Инициализируем wandb
    wandb_enabled = True
    try:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "pretrained_model": pretrained_model,
                "transformer_layers": transformer_layers,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "optimizer": optimizer_type.value,
                "weight_decay": weight_decay,
                "scheduler": scheduler_type.value,
                "warmup_steps": warmup_steps,
                "seed": seed,
                **vars(model_parameters),
            },
        )
    except Exception:
        wandb_enabled = False
        if verbose:
            print("Не удалось инициализировать wandb. Логи не будут отправляться.")

    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(output_dir, exist_ok=True)

    # Цикл по эпохам
    for epoch in range(start_epoch, epochs + 1):
        # Обучение за эпоху
        epoch_loss, global_step = train_one_epoch(
            pipe=pipe,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_type=scheduler_type,
            criterion=criterion,
            model_parameters=model_parameters,
            device_obj=device_obj,
            verbose=verbose,
            progress_bar=progress_bar,
            epoch=epoch,
            global_step=global_step,
            logging_steps=logging_steps,
            wandb_enabled=wandb_enabled,
        )

        # Сохраняем чекпоинт по шагам, если нужно
        if save_steps and global_step % save_steps < len(train_loader):
            meta_info = {
                "epoch": epoch,
                "global_step": global_step,
                "pretrained_model": pretrained_model,
                "transformer_layers": transformer_layers,
                **vars(model_parameters),
            }
            save_checkpoint(output_dir, f"step-{global_step}", pipe, optimizer, scheduler, meta_info, verbose)

        # Сохраняем чекпоинт после завершения эпохи
        meta_info_epoch = {
            "epoch": epoch,
            "global_step": global_step,
            "pretrained_model": pretrained_model,
            "transformer_layers": transformer_layers,
            **vars(model_parameters),
        }
        save_checkpoint(output_dir, f"epoch-{epoch}", pipe, optimizer, scheduler, meta_info_epoch, verbose)

        # Валидация
        metrics = validate_one_epoch(
            pipe=pipe,
            val_loader=val_loader,
            model_parameters=model_parameters,
            device_obj=device_obj,
            output_dir=output_dir,
            epoch=epoch,
            verbose=verbose,
            progress_bar=progress_bar,
            wandb_enabled=wandb_enabled,
        )

    # Сохраняем финальный чекпоинт
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_clf(str(final_dir / "classifier.safetensors"))
    torch.save(
        {
            "optimizer_state": optimizer.state_dict(),
            "meta": {
                "epoch": epochs,
                "pretrained_model": pretrained_model,
                "transformer_layers": transformer_layers,
                **vars(model_parameters),
            },
        },
        final_dir / "optimizer.pt",
    )
    if wandb_enabled:
        wandb.finish()


def evaluate(
    test_df: pd.DataFrame,
    pretrained_model: str,
    model_parameters: ModelParameters,
    transformer_layers: List[int],
    checkpoint_path: str,
    threshold: float = .5
    batch_size: int = 8,
    device: str = "cuda",
    output_csv: str = "eval_results.csv",
    verbose: bool = True,
    progress_bar: bool = True,
):
    """
    Функция для оценки модели на тестовом наборе:
    - Загружает модель из чекпоинта
    - Делает предсказания и вычисляет метрики
    - Сохраняет результаты в CSV
    - Печатает метрики при verbose=True
    """
    if "fp" not in test_df.columns or "label" not in test_df.columns:
        raise ValueError("test_df должен содержать колонки 'fp' и 'label'")

    device_obj = torch.device(device)
    pipe = prepare_model(pretrained_model, transformer_layers, model_parameters, device)
    pipe.load_clf(str(Path(checkpoint_path) / "classifier.safetensors"))
    pipe.eval()

    dataset = ImageLabelDataset(test_df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_image_label,
        drop_last=False,
    )

    all_probs: List[float] = []
    all_preds: List[int] = []
    all_labels: List[float] = []
    all_paths: List[str] = []

    total_batches = len(dataloader)
    iterator = dataloader
    if progress_bar:
        iterator = tqdm(dataloader, total=total_batches, desc="Evaluation", leave=False)

    with torch.no_grad():
        for images, labels, paths in iterator:
            batch_size_actual = len(images)
            probs_batch = []
            preds_batch = []
            for i in range(batch_size_actual):
                img = images[i]
                logit = pipe(img, model_parameters.prompt, model_parameters.t)
                prob = torch.sigmoid(logit).item()
                pred = 1 if prob >= threshold else 0
                probs_batch.append(prob)
                preds_batch.append(pred)

            all_probs.extend(probs_batch)
            all_preds.extend(preds_batch)
            all_labels.extend(labels.tolist())
            all_paths.extend(paths)

    # Сохраняем результаты в CSV
    df_out = pd.DataFrame({
        "fp": all_paths,
        "label": all_labels,
        "prob": all_probs,
        "pred": all_preds,
    })
    os.makedirs(Path(output_csv).parent, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    # Вычисляем метрики
    y_true = df_out["label"]
    y_pred = df_out["pred"]
    y_prob = df_out["prob"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    if verbose:
        print(f"Evaluation results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {auc:.4f}")
        print("  Confusion Matrix:")
        print(cm)
        print("  Classification Report:")
        print(report)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }