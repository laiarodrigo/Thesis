from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence


TRANSLATION_TASKS = {
    "translation",
    "translate_br2pt",
    "translate_pt2br",
}
CLASSIFICATION_TASKS = {
    "classification",
    "classify",
}


def normalize_task_kind(value: object) -> str:
    task = str(value or "").strip().casefold()
    if task in TRANSLATION_TASKS:
        return "translation"
    if task in CLASSIFICATION_TASKS:
        return "classification"
    raise ValueError(f"Unsupported task value for balanced sampling: {value!r}")


class FixedTaskMixSampler:
    """Yield fixed translation/classification mixtures per optimizer window."""

    def __init__(
        self,
        task_values: Sequence[object],
        *,
        window_size: int,
        translation_rows_per_window: int,
        classification_rows_per_window: int,
        seed: int,
    ) -> None:
        self.window_size = int(window_size)
        self.translation_rows_per_window = int(translation_rows_per_window)
        self.classification_rows_per_window = int(classification_rows_per_window)
        self.seed = int(seed)
        self.epoch = 0

        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.translation_rows_per_window <= 0:
            raise ValueError("translation_rows_per_window must be > 0")
        if self.classification_rows_per_window <= 0:
            raise ValueError("classification_rows_per_window must be > 0")
        if (
            self.translation_rows_per_window + self.classification_rows_per_window
            != self.window_size
        ):
            raise ValueError(
                "translation_rows_per_window + classification_rows_per_window "
                "must equal window_size"
            )

        self.indices_by_task: dict[str, list[int]] = {
            "translation": [],
            "classification": [],
        }
        for idx, task_value in enumerate(task_values):
            self.indices_by_task[normalize_task_kind(task_value)].append(idx)

        for task_kind, indices in self.indices_by_task.items():
            if not indices:
                raise ValueError(
                    f"Balanced sampling requires at least one {task_kind} row"
                )

        self.windows_per_epoch = max(
            math.ceil(
                len(self.indices_by_task["translation"])
                / self.translation_rows_per_window
            ),
            math.ceil(
                len(self.indices_by_task["classification"])
                / self.classification_rows_per_window
            ),
        )

    def __len__(self) -> int:
        return self.windows_per_epoch * self.window_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _take_rows(
        *,
        pool: list[int],
        position: int,
        count: int,
        rng: random.Random,
    ) -> tuple[list[int], int]:
        rows: list[int] = []
        while len(rows) < count:
            if position >= len(pool):
                rng.shuffle(pool)
                position = 0
            available = min(count - len(rows), len(pool) - position)
            rows.extend(pool[position : position + available])
            position += available
        return rows, position

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        pools = {
            task_kind: list(indices)
            for task_kind, indices in self.indices_by_task.items()
        }
        for pool in pools.values():
            rng.shuffle(pool)
        positions = {"translation": 0, "classification": 0}

        for _ in range(self.windows_per_epoch):
            translation_rows, positions["translation"] = self._take_rows(
                pool=pools["translation"],
                position=positions["translation"],
                count=self.translation_rows_per_window,
                rng=rng,
            )
            classification_rows, positions["classification"] = self._take_rows(
                pool=pools["classification"],
                position=positions["classification"],
                count=self.classification_rows_per_window,
                rng=rng,
            )
            window = [*translation_rows, *classification_rows]
            rng.shuffle(window)
            yield from window
