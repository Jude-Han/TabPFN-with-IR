import os

from tabpfn_ir.environment import load_project_dotenv


def test_load_project_dotenv_loads_tabpfn_token_without_returning_it(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("TABPFN_TOKEN=test-secret-value\n", encoding="utf-8")
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)

    loaded_path = load_project_dotenv(dotenv_path)

    assert loaded_path == dotenv_path.resolve()
    assert os.environ["TABPFN_TOKEN"] == "test-secret-value"
    assert "test-secret-value" not in str(loaded_path)


def test_load_project_dotenv_does_not_override_exported_token(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("TABPFN_TOKEN=file-value\n", encoding="utf-8")
    monkeypatch.setenv("TABPFN_TOKEN", "exported-value")

    load_project_dotenv(dotenv_path, override=False)

    assert os.environ["TABPFN_TOKEN"] == "exported-value"


def test_load_project_dotenv_ignores_a_missing_file(tmp_path):
    assert load_project_dotenv(tmp_path / ".env") is None
