from pathlib import Path


def test_no_banned_markers():
    root = Path(__file__).resolve().parents[1]
    banned = ["TODO", "todo", "pass", "NotImplementedError", "raise NotImplementedError", "placeholder", "stub"]
    files = list((root / "mrvp").glob("**/*.py")) + [root / "README.md"]
    offenders = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for marker in banned:
            if marker in text:
                offenders.append(f"{path.relative_to(root)}:{marker}")
    assert offenders == []
