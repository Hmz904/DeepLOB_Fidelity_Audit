import runpy


def test_author_archive_url_is_commit_pinned_and_size_guarded():
    ns = runpy.run_path("scripts/download_author_setup2.py")
    commit = ns["AUTHOR_DATA_COMMIT"]
    url = ns["URL"]
    assert commit == "d45844b022209bd9d7985de97076f2e80c5144dc"
    assert f"/{commit}/data/data.zip" in url
    assert "/master/" not in url
    assert ns["EXPECTED_ARCHIVE_BYTES"] == 56_278_154
