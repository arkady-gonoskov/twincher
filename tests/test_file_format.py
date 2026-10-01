# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""Tests of saving and loading twinchers (.twc files with metadata)."""

import os
import struct
import numpy as np
import pytest
import twincher
from helpers import to_np

def make_shuttle():
    SH = twincher.Shuttle(data_type="np", n_s=8, n_l=4, n_c=16, n_p=2, n_m=2)
    SH.set_rnd_t(3)
    SH.metadata["note"] = "test"
    return SH

def output(SH):
    SH.s[:] = 0.3
    SH.forward_inference()
    return to_np(SH.s)

@pytest.fixture
def saved(tmp_path):
    """A saved twincher: (file name, its content, the output of the twincher)."""
    SH = make_shuttle()
    file_name = str(tmp_path / "saved.twc")
    SH.save(file_name)
    return file_name, open(file_name, "rb").read(), output(SH)

def load(file_name, data_type="np"):
    SH = twincher.Shuttle(data_type=data_type)
    SH.load(file_name, n_c=16, n_p=2, n_m=2)
    return SH

def test_header(saved):
    _, content, _ = saved
    assert content[:8] == b"TWINCHER"
    assert struct.unpack("<I", content[8:12])[0] == 1 # format version

def test_round_trip(saved):
    file_name, _, reference = saved
    SH = load(file_name)
    assert SH.metadata == {"twincher_type": "untrained", "note": "test"}
    assert twincher.load_metadata(file_name) == SH.metadata
    assert (SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m) == (8, 4, 16, 2, 2)
    assert np.array_equal(output(SH), reference)

def test_load_with_other_data_types(saved, data_type):
    # a file written with float64 (CPU) can be loaded for float32 (GPU) and vice versa
    file_name, _, reference = saved
    SH = load(file_name, data_type)
    assert np.abs(output(SH) - reference).max() < 1e-6

def test_version_0_file(saved, tmp_path):
    # files written before the introduction of format versions have no magic string and
    # version: they start with the size of the metadata
    file_name, content, reference = saved
    old_file = str(tmp_path / "version_0.twc")
    open(old_file, "wb").write(content[12:])
    SH = load(old_file)
    assert SH.metadata == {"twincher_type": "untrained", "note": "test"}
    assert np.array_equal(output(SH), reference)

def test_newer_version_is_rejected(saved, tmp_path):
    _, content, _ = saved
    file_name = str(tmp_path / "future.twc")
    open(file_name, "wb").write(content[:8] + struct.pack("<I", 2) + content[12:])
    with pytest.raises(ValueError, match="please update twincher"):
        twincher.load_metadata(file_name)

@pytest.mark.parametrize("kind", ["empty", "garbage", "cut after magic string", "truncated metadata"])
def test_invalid_files_are_rejected(saved, tmp_path, kind):
    _, content, _ = saved
    invalid = {
        "empty": b"",
        "garbage": os.urandom(100),
        "cut after magic string": content[:10],
        "truncated metadata": content[:30],
    }[kind]
    file_name = str(tmp_path / "invalid.twc")
    open(file_name, "wb").write(invalid)
    with pytest.raises(ValueError):
        twincher.load_metadata(file_name)
    with pytest.raises(ValueError):
        load(file_name)

def test_truncated_core_data_is_rejected(saved, tmp_path):
    _, content, _ = saved
    file_name = str(tmp_path / "truncated.twc")
    open(file_name, "wb").write(content[:-20])
    with pytest.raises(RuntimeError, match="truncated"):
        load(file_name)

def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        twincher.load_metadata(str(tmp_path / "missing.twc"))
