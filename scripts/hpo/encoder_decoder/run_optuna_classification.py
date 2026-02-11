#!/usr/bin/env python3
from _optuna_runner import run_optuna


if __name__ == "__main__":
    run_optuna("hpo/encoder_decoder/classification_search_space.yaml")
