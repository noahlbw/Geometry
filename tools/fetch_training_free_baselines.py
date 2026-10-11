"""Acquire official OVSS baseline sources without installing or executing them."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "third_party" / "training_free_ovss_20261010"
HEADERS = {"User-Agent": "Geometry-baseline-source-review", "Accept": "application/vnd.github+json"}


def request(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=45) as response:
        return response.read()


def github_json(path: str):
    return json.loads(request("https://api.github.com/" + path))


def readme(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}/readme"
    headers = {**HEADERS, "Accept": "application/vnd.github.raw+json"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
        return response.read().decode("utf-8", "replace")


def search(query: str):
    result = github_json("search/repositories?" + urllib.parse.urlencode({"q": query, "per_page": 100, "sort": "updated"}))
    keys = ("full_name", "description", "homepage", "created_at", "pushed_at", "size", "default_branch")
    return {"query": query, "total_count": result["total_count"], "items": [{k: item.get(k) for k in keys} for item in result["items"]]}


def paper(identifier: str):
    url = "https://arxiv.org/abs/" + identifier
    content = request(url).decode("utf-8", "replace")
    strip = lambda value: html.unescape(re.sub(r"<[^>]+>", " ", value)).strip()
    fields = {
        "title": r'<h1 class="title mathjax">(.*?)</h1>',
        "date": r'<div class="dateline">(.*?)</div>',
        "abstract": r'<blockquote class="abstract mathjax">(.*?)</blockquote>',
        "journal_ref": r'<td class="tablecell jref">(.*?)</td>',
    }
    result = {"url": url}
    for key, expression in fields.items():
        match = re.search(expression, content, re.S)
        result[key] = strip(match.group(1)) if match else None
    return result


def restore_inheritance(target: Path):
    # TemporaryDirectory has a private Windows ACL; a moved checkout must
    # inherit the workspace ACL so both the user and sandbox can read it.
    if os.name == "nt":
        subprocess.run(["icacls.exe", str(target), "/reset", "/T", "/C", "/Q"],
                       check=True, capture_output=True, timeout=60)


def git_environment():
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", GIT_LFS_SKIP_SMUDGE="1")
    proxies = urllib.request.getproxies()
    for protocol in ("https", "http"):
        if proxies.get(protocol):
            env[protocol.upper() + "_PROXY"] = proxies[protocol]
    return env


def run_git(args: list[str], timeout: int = 45):
    scoped_safety = []
    if "-C" in args:
        scoped_safety = ["-c", "safe.directory=" + Path(args[args.index("-C") + 1]).resolve().as_posix()]
    return subprocess.run(
        ["git", "-c", "http.sslBackend=openssl", "-c", "http.version=HTTP/1.1", *scoped_safety, *args],
        env=git_environment(), text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=timeout,
    )


def inventory(source: Path):
    files = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file() and ".git" not in p.relative_to(source).parts)
    return {
        "file_count": len(files),
        "python_files": sum(p.endswith(".py") for p in files),
        "entrypoint_candidates": [p for p in files if re.search(r"(^|/)(eval[^/]*|test[^/]*|demo[^/]*|inference[^/]*|segment[^/]*)\.py$", p, re.I)][:35],
        "environment_files": [p for p in files if re.search(r"(^|/)(requirements[^/]*|environment[^/]*|setup\.py|pyproject\.toml)$", p, re.I)][:20],
        "config_files": [p for p in files if p.endswith((".yaml", ".yml")) or ("config" in p.lower() and p.endswith(".py"))][:30],
        "license_files": [p for p in files if re.search(r"(^|/)(license[^/]*|copying[^/]*)$", p, re.I)],
    }


def acquire(repo: str):
    target = ROOT / repo.rsplit("/", 1)[-1]
    record = {"repository": repo, "url": f"https://github.com/{repo}", "local_path": str(target), "evaluation_status": "not_run"}
    try:
        if target.exists():
            provenance = target / "SOURCE_PROVENANCE.json"
            if provenance.exists():
                record.update(json.loads(provenance.read_text(encoding="utf-8")))
            elif (target / ".git").exists():
                remote = run_git(["-C", str(target), "remote", "get-url", "origin"])
                if remote.returncode or remote.stdout.strip().rstrip("/").removesuffix(".git") != record["url"]:
                    raise RuntimeError("Existing target belongs to a different repository; preserved")
                head = run_git(["-C", str(target), "rev-parse", "HEAD"])
                record.update(acquisition="existing_git_clone", commit=head.stdout.strip())
            else:
                raise RuntimeError("Existing target has no verified source provenance; preserved")
            restore_inheritance(target)
            record["inventory"] = inventory(target)
            return record
        metadata = github_json(f"repos/{repo}")
        branch = metadata["default_branch"]
        commit = github_json(f"repos/{repo}/commits/{urllib.parse.quote(branch, safe='')}")["sha"]
        record.update(commit=commit, branch=branch, description=metadata.get("description"),
                      homepage=metadata.get("homepage"), license=(metadata.get("license") or {}).get("spdx_id"),
                      retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat())
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="source-fetch-", dir=ROOT) as temporary:
            stage = Path(temporary) / "checkout"
            try:
                clone = run_git(["clone", "--depth", "1", "--single-branch", "--branch", branch,
                                 record["url"] + ".git", str(stage)], timeout=40)
                cloned = clone.returncode == 0
            except subprocess.TimeoutExpired:
                cloned = False
            if cloned:
                head = run_git(["-C", str(stage), "rev-parse", "HEAD"])
                if head.returncode:
                    raise RuntimeError("Clone HEAD could not be resolved")
                record.update(acquisition="git_clone_depth1", commit=head.stdout.strip())
                shutil.move(str(stage), str(target))
            else:
                archive_path = Path(temporary) / "source.zip"
                archive_path.write_bytes(request(f"https://codeload.github.com/{repo}/zip/{commit}"))
                extracted = Path(temporary) / "extracted"
                with zipfile.ZipFile(archive_path) as archive:
                    base = extracted.resolve()
                    for member in archive.infolist():
                        resolved = (extracted / member.filename).resolve()
                        if not resolved.is_relative_to(base):
                            raise RuntimeError("Source archive has a path outside its extraction directory")
                    archive.extractall(extracted)
                folders = [p for p in extracted.iterdir() if p.is_dir()]
                if len(folders) != 1:
                    raise RuntimeError("Unexpected source archive root structure")
                shutil.move(str(folders[0]), str(target))
                record["acquisition"] = "official_commit_zip_snapshot"
        restore_inheritance(target)
        record["inventory"] = inventory(target)
        (target / "SOURCE_PROVENANCE.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return record
    except Exception as exc:
        record.update(acquisition="failed", error=f"{type(exc).__name__}: {exc}")
        return record


def audit(repo: str):
    target = ROOT / repo.rsplit("/", 1)[-1]
    record = json.loads((target / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    head = run_git(["-C", str(target), "rev-parse", "HEAD"])
    changes = run_git(["-C", str(target), "diff", "--name-only", "HEAD"])
    if head.returncode or changes.returncode:
        raise RuntimeError("Could not inspect downloaded Git checkout")
    record.update(commit_verified=head.stdout.strip() == record["commit"],
                  tracked_source_changes=changes.stdout.splitlines())
    if not record["commit_verified"] or record["tracked_source_changes"]:
        raise RuntimeError("Downloaded checkout differs from its recorded source version")
    record["inventory"] = inventory(target)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("search", "readme", "paper", "acquire", "audit"))
    parser.add_argument("items", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chars", type=int, default=6000)
    args = parser.parse_args()
    def perform(item):
        try:
            if args.action == "search":
                return search(item)
            if args.action == "readme":
                return {"repository": item, "readme_excerpt": readme(item)[:args.chars]}
            if args.action == "paper":
                return paper(item)
            if args.action == "audit":
                return audit(item)
            return acquire(item)
        except Exception as exc:
            return {"item": item, "error": f"{type(exc).__name__}: {exc}"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(perform, args.items))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
