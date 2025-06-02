import os
from pathlib import Path
from typing import Union, Dict, Any

import torch
from PIL import Image
from transformers import PreTrainedTokenizer

from .pipeline import SanaClassifierPipeline, ModelParameters
from .train import prepare_model


class SanaClassifierInference:
    def __init__(self, checkpoint_dir: Union[str, Path], device: str = "cuda"):
        """
        Инициализация:
        - Загружает метаинформацию из optimizer.pt (поле 'meta')
        - Воссоздает ModelParameters из meta
        - Создает SanaClassifierPipeline через prepare_model
        - Загружает веса классификатора из classifier.safetensors

        Args:
            checkpoint_dir (str or Path): путь к директории чекпоинта,
                содержащей файлы classifier.safetensors и optimizer.pt
            device (str): устройство для инференса ('cuda' или 'cpu')
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        optimizer_pt_path = checkpoint_dir / "optimizer.pt"
        if not optimizer_pt_path.exists():
            raise FileNotFoundError(f"optimizer.pt not found in {checkpoint_dir}")
        checkpoint = torch.load(str(optimizer_pt_path), map_location=device)

        if "meta" not in checkpoint:
            raise KeyError(f"'meta' not found in {optimizer_pt_path}")
        meta: Dict[str, Any] = checkpoint["meta"]

        if "pretrained_model" not in meta or "transformer_layers" not in meta:
            raise KeyError("В meta отсутствуют 'pretrained_model' или 'transformer_layers'")

        self.pretrained_model: str = meta.pop("pretrained_model")
        self.transformer_layers: list = meta.pop("transformer_layers")

        try:
            self.model_parameters = ModelParameters(**meta)
        except Exception as e:
            raise ValueError(f"Не удалось создать ModelParameters из meta: {e}")

        self.device = device
        self.pipe: SanaClassifierPipeline = prepare_model(
            pretrained_model=self.pretrained_model,
            transformer_layers=self.transformer_layers,
            model_parameters=self.model_parameters,
            device=self.device,
        )

        clf_path = checkpoint_dir / "classifier.safetensors"
        if not clf_path.exists():
            raise FileNotFoundError(f"classifier.safetensors not found in {checkpoint_dir}")
        self.pipe.load_clf(str(clf_path))
        self.pipe.eval()

    def __call__(self, img: Union[str, Image.Image], threshold: float = 0.5) -> Dict[str, Any]:
        """
        Выполняет инференс на одном изображении.

        Args:
            img (str or PIL.Image.Image): путь к изображению или объект PIL.Image
            threshold (float): порог для предсказания класса (по умолчанию 0.5)

        Returns:
            dict: {
                "probability": float,  # вероятность положительного класса
                "prediction": int      # 1, если probability >= threshold, иначе 0
            }
        """
        # Загрузка или валидация изображения
        if isinstance(img, str):
            if not os.path.exists(img):
                raise FileNotFoundError(f"Изображение не найдено: {img}")
            image = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            image = img.convert("RGB")
        else:
            raise TypeError("img должен быть либо путем к файлу (str), либо PIL.Image.Image")

        # Получаем логит из пайплайна
        # pipeline ожидает два аргумента: изображение и параметры (prompt, t)
        logit = self.pipe(image, self.model_parameters.prompt, self.model_parameters.t)
        prob = torch.sigmoid(logit).item()
        pred = 1 if prob >= threshold else 0

        return {"probability": prob, "prediction": pred}