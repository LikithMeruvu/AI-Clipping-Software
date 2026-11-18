import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from clipper.config import Settings
from clipper.web import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", output_dir=tmp_path / "out",
                        api_token=SecretStr("s3cret"))
    return TestClient(create_app(settings)), tmp_path


AUTH = {"Authorization": "Bearer s3cret"}


def test_health_is_public_and_lists_styles(client):
    c, _ = client
    body = c.get("/api/health").json()
    assert body["ok"] and "hormozi" in body["styles"] and "9:16" in body["aspects"]


def test_api_requires_token(client):
    c, _ = client
    assert c.get("/api/jobs").status_code == 401
    assert c.get("/api/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/jobs", headers=AUTH).status_code == 200


def test_job_creation_validates_input(client):
    c, _ = client
    assert c.post("/api/jobs", headers=AUTH, data={}).status_code == 422
    bad = json.dumps({"min_seconds": 50, "max_seconds": 20})
    assert c.post("/api/jobs", headers=AUTH, data={"url": "https://youtu.be/abcdefghij", "options": bad}).status_code == 422
    bad_style = json.dumps({"style": "nope"})
    assert c.post("/api/jobs", headers=AUTH, data={"url": "https://youtu.be/abcdefghij", "options": bad_style}).status_code == 422


def test_upload_rejects_unsupported_extension(client):
    c, _ = client
    r = c.post("/api/jobs", headers=AUTH, files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415


def test_file_endpoint_blocks_traversal_and_unknown_jobs(client):
    c, _ = client
    assert c.get("/api/jobs/abcdef123456/files/..%2Fjob.json", headers=AUTH).status_code == 404
    assert c.get("/api/jobs/../files/x", headers=AUTH).status_code in (404, 405)
    assert c.get("/api/jobs/abcdef123456", headers=AUTH).status_code == 404


def test_non_youtube_url_job_fails_cleanly(client):
    import time
    c, _ = client
    r = c.post("/api/jobs", headers=AUTH, data={"url": "https://evil.example/x"})
    assert r.status_code == 201
    jid = r.json()["id"]
    for _ in range(50):
        job = c.get(f"/api/jobs/{jid}", headers=AUTH).json()
        if job["status"] in ("failed", "done"):
            break
        time.sleep(0.1)
    assert job["status"] == "failed" and "origin_path" not in job
