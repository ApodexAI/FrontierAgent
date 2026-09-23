import hashlib
import io
import json
import tarfile

import pytest
from setup_release import verify_loaded_image_identity, verify_oci_image_identity

CONFIG = "sha256:" + "a" * 64


def fixture(*, config=CONFIG, media_type="application/vnd.oci.image.manifest.v1+json",
            corrupt=False, missing=False, symlink=False, oversized=False):
    raw = json.dumps({"schemaVersion": 2, "mediaType": media_type,
                      "config": {"digest": config}}).encode()
    identity = "sha256:" + hashlib.sha256(raw).hexdigest()
    if corrupt:
        raw += b" "
    if oversized:
        raw += b" " * (1024 * 1024)
        identity = "sha256:" + hashlib.sha256(raw).hexdigest()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("other" if missing else "blobs/sha256/" + identity[7:])
        if symlink:
            member.type = tarfile.SYMTYPE
            member.linkname = "/should-not-be-read"
        else:
            member.size = len(raw)
        archive.addfile(member, None if symlink else io.BytesIO(raw))
    stream.seek(0)
    return stream, identity


def test_classic_identity_needs_no_decompression(tmp_path):
    verify_loaded_image_identity(tmp_path / "not-opened", CONFIG, CONFIG)


@pytest.mark.parametrize("media_type", ["application/vnd.oci.image.manifest.v1+json",
                                      "application/vnd.docker.distribution.manifest.v2+json"])
def test_containerd_identity_is_bound_to_published_config(media_type):
    stream, identity = fixture(media_type=media_type)
    assert identity != CONFIG
    verify_oci_image_identity(stream, identity, CONFIG)


@pytest.mark.parametrize("kwargs,expected", [
    ({"config": "sha256:" + "b" * 64}, "configuration does not match"),
    ({"corrupt": True}, "digest mismatch"),
    ({"missing": True}, "no matching manifest"),
    ({"symlink": True}, "invalid OCI image manifest entry"),
    ({"oversized": True}, "invalid OCI image manifest entry"),
    ({"media_type": "application/vnd.oci.image.index.v1+json"}, "configuration does not match"),
])
def test_identity_mismatch_still_fails_closed(kwargs, expected):
    stream, identity = fixture(**kwargs)
    with pytest.raises(SystemExit, match=expected):
        verify_oci_image_identity(stream, identity, CONFIG)


def test_invalid_digest_rejected_before_reading():
    with pytest.raises(SystemExit, match="invalid image identity"):
        verify_oci_image_identity(io.BytesIO(), "../../wrong", CONFIG)
