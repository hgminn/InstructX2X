# instruct-pix2pix 스타일을 그대로 유지하면서
# 1) CSV 1 개(instructions.csv + split 열) 사용
# 2) 이미지 폴더를  past/  ,  current/  두 곳으로 분리
# 외에는 기존과 동일한 동작

from __future__ import annotations
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


def _to_tensor(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img, np.float32)) * (2.0 / 255.0) - 1.0
    return rearrange(t, "h w -> 1 h w").repeat(3, 1, 1)


class EditDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        split: Literal["train", "val", "test"] = "train",
        *,
        min_resize_res: int = 512,
        max_resize_res: int = 512,
        crop_res: int = 512,
        rot_degree: float = 2.0,
        flip_prob: float = 0.0,
    ):
        self.path = Path(path)
        df = pd.read_csv(self.path / "instructions.csv")
        self.data = df[df["split"] == split].reset_index(drop=True)

        self.dir_current = self.path / "current"
        self.dir_past = self.path / "past"

        self.min_r = min_resize_res
        self.max_r = max_resize_res
        self.aug = T.Compose(
            [
                T.RandomRotation(rot_degree),
                T.CenterCrop(crop_res),
                T.RandomHorizontalFlip(flip_prob),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.data.iloc[i]
        res = torch.randint(self.min_r, self.max_r + 1, ()).item()

        img_cur = Image.open(self.dir_current / row["current_dicom_id"]).convert("L").resize(
            (res, res), Image.Resampling.LANCZOS
        )
        img_pas = Image.open(self.dir_past / row["past_dicom_id"]).convert("L").resize(
            (res, res), Image.Resampling.LANCZOS
        )

        cur_t, past_t = map(_to_tensor, (img_cur, img_pas))
        cur_t, past_t = self.aug(torch.cat((cur_t, past_t))).chunk(2)

        return dict(edited=cur_t, edit=dict(c_concat=past_t, c_crossattn=row["instruction"]))


class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str | Path,
        split: Literal["train", "val", "test"] = "test",
        *,
        res: int = 512,
    ):
        self.path = Path(path)
        df = pd.read_csv(self.path / "instructions.csv")
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.dir_past = self.path / "past"
        self.res = res

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.data.iloc[i]
        img = Image.open(self.dir_past / row["past_dicom_id"]).convert("L").resize(
            (self.res, self.res), Image.Resampling.LANCZOS
        )
        img_t = _to_tensor(img)
        return dict(image_0=img_t, input_prompt=[""], edit=row["instruction"], output_prompt=[""])