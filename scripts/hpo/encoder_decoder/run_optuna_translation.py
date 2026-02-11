#!/usr/bin/env python3
from _optuna_runner import run_optuna


if __name__ == "__main__":
    run_optuna("hpo/encoder_decoder/translation_search_space.yaml")
