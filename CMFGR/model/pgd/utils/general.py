
import logging
LOGGER = logging.getLogger("PGD")

import os
import sys
import yaml
import math

def check_version(current: str, minimum: str = '0.0.0', name: str = 'version', pinned: bool = False, hard: bool = False, verbose: bool = False):
	# YOLOv5 version check
	from packaging import version
	current, minimum = version.parse(current), version.parse(minimum)
	result = (current == minimum) if pinned else (current >= minimum)
	if hard:
		assert result, f'WARNING: {name} {minimum} is required, but {current} is installed.'
	if verbose and not result:
		print(f'WARNING: {name} {minimum} is required, but {current} is installed.')
	return result

def check_yaml(file, suffix=('.yaml', '.yml')):
	# Search/download YAML file (if necessary) and return path, checking suffix
	if isinstance(file, (list, tuple)):
		return [check_yaml(f, suffix) for f in file]
	if not isinstance(file, str):
		return file
	if os.path.splitext(file)[-1] not in suffix:
		raise ValueError(f'File {file} does not have a valid YAML extension {suffix}')
	return file

def make_divisible(x, divisor):
	# Returns x evenly divisible by divisor
	return math.ceil(x / divisor) * divisor

def print_args(args: dict):
	# Print argparse arguments
	for k, v in args.items():
		print(f'{k}: {v}')
