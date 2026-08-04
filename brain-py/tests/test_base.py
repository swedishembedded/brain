# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Unit tests for the transport-agnostic bits of `brain_py.base` that don't need
a brain process or a bus: `EXIT_SKIP`/`skip()` and `BrainError`.
"""
from __future__ import annotations

import pytest

from brain_py.base import EXIT_SKIP, BrainError, skip


def test_exit_skip_is_the_automake_skip_convention():
    assert EXIT_SKIP == 77


def test_skip_prints_the_reason_and_exits_with_exit_skip(capsys):
    with pytest.raises(SystemExit) as exc_info:
        skip("no GPU available")
    assert exc_info.value.code == EXIT_SKIP
    assert "SKIP: no GPU available" in capsys.readouterr().err


def test_brain_error_formats_with_and_without_a_name():
    assert str(BrainError("boom")) == "boom"
    assert str(BrainError("boom", name="org.freedesktop.DBus.Error.Failed")) == "[org.freedesktop.DBus.Error.Failed] boom"
    assert isinstance(BrainError("boom"), RuntimeError)
