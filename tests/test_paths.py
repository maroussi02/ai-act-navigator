"""UNIT TEST: data paths don't depend on the folder a command is started from."""

from ai_act_navigator import paths


def test_data_dir_is_inside_the_project():
    assert paths.PROJECT_ROOT.joinpath("pyproject.toml").exists()
    assert paths.DATA_DIR == paths.PROJECT_ROOT / "data"
