from adaptive_vap_space.paths import relpath, resolve_under


def test_relpath_and_resolve(tmp_path):
    root = tmp_path / "root"
    p = root / "a" / "b.txt"
    p.parent.mkdir(parents=True)
    p.write_text("x")
    r = relpath(p, root)
    assert r == "a/b.txt"
    assert resolve_under(root, r) == p
