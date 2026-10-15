# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Fixture: a seed flag with no default is not a literal."""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=None)
