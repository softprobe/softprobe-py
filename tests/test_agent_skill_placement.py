from pathlib import Path


def test_developer_skills_live_in_the_central_workspace() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    local_skill_directories = (
        ".agents/skills",
        ".cursor/skills",
        ".agent/skills",
    )
    present_directories = [
        relative_path
        for relative_path in local_skill_directories
        if (repository_root / relative_path).exists()
    ]

    assert not present_directories, (
        "Developer skills must live in the central mono repo softprobe/dev at "
        ".agents/skills/ because agents run from that workspace. Remove: "
        f"{present_directories}"
    )
