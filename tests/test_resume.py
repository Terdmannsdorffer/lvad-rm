"""A PPO run that is interrupted or paused ends with exactly the parameters of one that is not."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import phase7_common  # noqa: E402
import phase7_train  # noqa: E402

KW = dict(n_envs=16, gamma=0.95, ent_coef=0.003)
BEATS = 16 * 25 * 14  # 14 updates: the second episode starts at update 13


def run(tmp, kind="last", lam=0.5):
    phase7_train.train("unknown", lam, kind, 10, 1.0, 0, BEATS, tmp / "runs", **KW)


def result(tmp):
    d = next((tmp / "runs").iterdir())
    log = json.loads((d / "train_log.json").read_text())
    return (d / "params.msgpack").read_bytes(), [{k: v for k, v in x.items() if k != "seconds"} for x in log], json.loads((d / "config.json").read_text())


@pytest.fixture
def place(tmp_path, monkeypatch):
    monkeypatch.setattr(phase7_train, "CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(phase7_train, "CKPT_EVERY", 5)
    monkeypatch.setattr(phase7_train, "PAUSE", tmp_path / "PAUSE")
    monkeypatch.setattr(phase7_common, "PAUSE", tmp_path / "PAUSE")
    return tmp_path


@pytest.mark.parametrize("kind,lam", [("last", 0.5), ("filter", 0.5)])  # naive, window and press also pass (slow)
def test_interrupted_run_is_identical(place, monkeypatch, kind, lam):
    run(place / "a", kind, lam)

    saves, original = [], phase7_train.save_checkpoint

    def crash_after_second_save(*a):
        original(*a)
        saves.append(1)
        if len(saves) == 2:
            raise RuntimeError("simulated GPU failure")

    monkeypatch.setattr(phase7_train, "save_checkpoint", crash_after_second_save)
    with pytest.raises(RuntimeError, match="simulated"):
        run(place / "b", kind, lam)
    assert len(list((place / "ckpt").glob("*.npz"))) == 1
    assert not list((place / "b").glob("runs/*/params.msgpack"))
    monkeypatch.setattr(phase7_train, "save_checkpoint", original)
    run(place / "b", kind, lam)
    assert result(place / "a") == result(place / "b")
    assert not list((place / "ckpt").iterdir())  # removed when the run ends


def test_pause_saves_and_leaves(place):
    run(place / "a")
    (place / "PAUSE").touch()
    with pytest.raises(SystemExit) as e:
        run(place / "b")
    assert e.value.code == phase7_common.EXIT_PAUSED
    (place / "PAUSE").unlink()
    run(place / "b")
    assert result(place / "a") == result(place / "b")


def test_pause_in_the_middle(place, monkeypatch):
    run(place / "a")
    original, n = phase7_train.make_ppo, [0]

    def make(*a):
        init, update, net = original(*a)

        def counted(ts):
            n[0] += 1
            if n[0] == 3:
                (place / "PAUSE").touch()
            return update(ts)

        return init, counted, net

    monkeypatch.setattr(phase7_train, "make_ppo", make)
    with pytest.raises(SystemExit):
        run(place / "b")
    assert n[0] == 3  # it leaves right after the update in which the pause was requested
    (place / "PAUSE").unlink()
    run(place / "b")
    assert result(place / "a") == result(place / "b")


def test_unusable_checkpoints_are_ignored(place, capsys):
    (place / "ckpt").mkdir()
    name = phase7_train.run_name("unknown", 0.5, "last", 10, 1.0, 0)
    (place / "ckpt" / f"runs_{name}.npz").write_bytes(b"truncated")
    run(place / "a")
    assert "checkpoint ignored" in capsys.readouterr().out
    assert result(place / "a")[2]["kind"] == "last"


def test_atomic_write_replaces_and_leaves_no_temporary(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("old")
    phase7_common.write_atomic(p, "new")
    assert p.read_text() == "new"
    assert [q.name for q in tmp_path.iterdir()] == ["x.json"]
