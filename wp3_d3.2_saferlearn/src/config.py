#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : config.py
# Author            : hargathor <3949704+hargathor@users.noreply.github.com>
# Date              : 04.04.2019
# Last Modified Date: 04.04.2019
# Last Modified By  : hargathor <3949704+hargathor@users.noreply.github.com>
"""Config module"""

import os


class Config(object):
    """Config class"""

    def __init__(self) -> None:
        pass

    def get_property(self, property_name):
        if property_name not in self._config.keys():  # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]

    BIND_PORT: str = os.environ.get("SAFERLEARN_PORT") or "5000"
    BIND_HOST: str = os.environ.get("SAFERLEARN_HOST") or "0.0.0.0"
    DEBUG: bool = True
    NB_COMPUTING_PARTIES: int = 3
    DATASETS_PATH: str = "/app_saferlearn/input-data/"
    WORKER_DB_PATH: str = "./worker.db"
    KAFKA_HOST: str = os.environ.get("KAFKA_HOST") or "kafka"
    KAFKA_PORT: int = os.environ.get("KAFKA_PORT") or 9092
    # Default to trained_nets_gpu if MODELS_PATH env var not set
    _default_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_nets_gpu")
    MODELS_PATH = os.environ.get("MODELS_PATH", _default_models_path)
