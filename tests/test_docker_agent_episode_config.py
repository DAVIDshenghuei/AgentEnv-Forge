from pathlib import Path

import agentenv_forge.runner as runner_module


def test_non_root_posix_identity_matches_workspace_owner_without_chown(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner_module.os, "getuid", lambda: 1234, raising=False)
    monkeypatch.setattr(runner_module.os, "getgid", lambda: 5678, raising=False)
    monkeypatch.setattr(
        runner_module.os,
        "chown",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-root workspace must not be chowned")
        ),
        raising=False,
    )

    assert runner_module._prepare_docker_workspace(tmp_path) == "1234:5678"


def test_non_root_posix_root_group_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module.os, "getuid", lambda: 1234, raising=False)
    monkeypatch.setattr(runner_module.os, "getgid", lambda: 0, raising=False)

    try:
        runner_module._prepare_docker_workspace(tmp_path)
    except ValueError as error:
        assert str(error) == "sandbox identity is unavailable"
    else:
        raise AssertionError("root group must be rejected")


def test_root_posix_identity_chowns_new_workspace_tree_to_fixed_non_root(
    tmp_path, monkeypatch
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    artifact = nested / "input.txt"
    artifact.write_text("input", encoding="utf-8")
    calls: list[tuple[Path, int, int, bool]] = []

    monkeypatch.setattr(runner_module.os, "getuid", lambda: 0, raising=False)
    monkeypatch.setattr(runner_module.os, "getgid", lambda: 0, raising=False)

    def record_chown(path, uid, gid, *, follow_symlinks):
        calls.append((Path(path), uid, gid, follow_symlinks))

    monkeypatch.setattr(runner_module.os, "chown", record_chown, raising=False)

    assert runner_module._prepare_docker_workspace(tmp_path) == "10001:10001"
    assert {call[0] for call in calls} == {tmp_path, nested, artifact}
    assert all(call[1:] == (10001, 10001, False) for call in calls)


def test_platform_without_posix_identity_uses_image_non_root_user(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delattr(runner_module.os, "getuid", raising=False)
    monkeypatch.delattr(runner_module.os, "getgid", raising=False)

    assert runner_module._prepare_docker_workspace(tmp_path) == "10001:10001"
