# -*- coding: utf-8 -*-
import os
import socket
import time
import json
import re
from datetime import datetime


def _safe_name(value):
    value = str(value).replace(":", "")
    value = value.replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    return value.strip("-")


def get_model_config_tag(model_params):
    if isinstance(model_params, str):
        try:
            model_params = json.loads(model_params)
        except json.JSONDecodeError:
            return _safe_name(model_params)[:120]

    if not isinstance(model_params, dict):
        return ""

    key_aliases = [
        ("batch_size", "bs"),
        ("d_model", "dm"),
        ("d_ff", "df"),
        ("num_epochs", "ep"),
        ("seq_len", "seq"),
        ("patch_size", "patch"),
        ("stride", "stride"),
        ("mask_ratio", "mask"),
        ("r", "r"),
        ("enc_in_time", "cin"),
        ("main_device", "main"),
        ("llm_device", "llm"),
    ]
    parts = []
    for key, alias in key_aliases:
        if key in model_params:
            parts.append(f"{alias}{_safe_name(model_params[key])}")
    return "_".join(parts)


def get_unique_file_suffix(descriptor=None):
    """
    Generate a log file name suffix that includes the following information:

    - Hostname
    - The current local time in a human-readable format
    - PID (process identifier) of the process

    Return:
    str: The name of the generated log file, in the format '.YYYYMMDD_HHMMSS.descriptor.hostname.pid.csv'

    For example, if the host name is' myhost ', the current timestamp is 1631655702, and the current process ID is 12345
    The returned file name may be '.1631655702.myhost.12345.csv'.
    """
    # Get Host Name
    hostname = socket.gethostname()

    # Get local wall-clock time instead of Unix epoch seconds
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")

    # Obtain the PID (process identifier) of the process
    pid = os.getpid()

    # Build file name
    descriptor_part = f".{_safe_name(descriptor)}" if descriptor else ""
    log_filename = f".{timestamp}{descriptor_part}.{hostname}.{pid}.csv"
    return log_filename
