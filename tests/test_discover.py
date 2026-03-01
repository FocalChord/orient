"""Tests for JPEG discovery."""

import pytest

from orient._discover import discover_jpegs


class TestDiscoverJpegs:
    def test_finds_jpg_files(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.jpeg").write_bytes(b"fake")
        (tmp_path / "c.txt").write_bytes(b"fake")

        result = discover_jpegs(tmp_path)
        names = [p.name for p in result]
        assert names == ["a.jpg", "b.jpeg"]

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "A.JPG").write_bytes(b"fake")
        (tmp_path / "B.Jpeg").write_bytes(b"fake")

        result = discover_jpegs(tmp_path)
        assert len(result) == 2

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.jpg").write_bytes(b"fake")
        (sub / "deep.jpg").write_bytes(b"fake")

        result = discover_jpegs(tmp_path, recursive=True)
        names = [p.name for p in result]
        assert "top.jpg" in names
        assert "deep.jpg" in names

    def test_non_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.jpg").write_bytes(b"fake")
        (sub / "deep.jpg").write_bytes(b"fake")

        result = discover_jpegs(tmp_path, recursive=False)
        names = [p.name for p in result]
        assert names == ["top.jpg"]

    def test_sorted_by_name(self, tmp_path):
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).write_bytes(b"fake")

        result = discover_jpegs(tmp_path)
        assert [p.name for p in result] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_empty_dir(self, tmp_path):
        assert discover_jpegs(tmp_path) == []

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"fake")
        with pytest.raises(NotADirectoryError):
            discover_jpegs(f)

    def test_nonexistent_path(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            discover_jpegs(tmp_path / "nope")
