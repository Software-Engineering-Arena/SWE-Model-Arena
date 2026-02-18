# References for model evaluation metrics:
# - Chatbot Arena: https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH
# - Evalica: https://github.com/dustalov/evalica/blob/master/Chatbot-Arena.ipynb

import asyncio
import concurrent.futures
import dotenv
import evalica
import gitlab
import httpx
import io
import json
import markdown as md_lib
import os
import random
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
import warnings

import gradio as gr
import pandas as pd

from datetime import datetime
from github import Auth, Github
from opencode_ai import AsyncOpencode
from urllib.parse import urlparse
from gradio_leaderboard import Leaderboard, ColumnFilter
from huggingface_hub import upload_file, hf_hub_download, HfApi
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment & constants
# ---------------------------------------------------------------------------

dotenv.load_dotenv(override=True)

# OpenAI client (guardrail only — models use opencode)
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = "https://openrouter.ai/api/v1"
openai_client = OpenAI(api_key=api_key, base_url=base_url)

# Hugging Face repository names for data storage
LEADERBOARD_REPO = "SWE-Arena/leaderboard_data"
VOTE_REPO = "SWE-Arena/vote_data"
CONVERSATION_REPO = "SWE-Arena/conversation_data"
MODEL_REPO = "SWE-Arena/model_data"
LEADERBOARD_FILE = "model_arena"

# Per-model timeout in seconds (how long one agent attempt can run)
AGENT_TIMEOUT = 300
# Total timeout for the entire battle (including all retries)
BATTLE_TIMEOUT = 600

# Leaderboard update time frame in days
LEADERBOARD_UPDATE_TIME_FRAME_DAYS = 365

# Hint string constant
SHOW_HINT_STRING = True
HINT_STRING = "Once signed in, your votes will be recorded securely."

# System prompt sent to every agent at the start of a battle.
# NOTE: the agent's concrete workspace path is appended by build_prompt() at
# runtime so the agent knows exactly where it may operate.
SYSTEM_PREFIX = (
    "You are an expert software engineer. "
    "The user will give you a task — follow their instructions precisely and completely. "
    "Do exactly what is asked: no more, no less. "
    "If the task involves writing or modifying code, produce clean, correct, and working code. "
    "If the task involves debugging, identify and fix the root cause. "
    "If the task involves explaining, be clear and concise. "
    "WORKSPACE CONSTRAINT: You have been given a dedicated workspace directory (see below). "
    "ALL file operations (read, write, create, modify, execute) must stay within that directory. "
    "You may use either relative paths (e.g. './src/foo.py') or absolute paths that are "
    "inside your workspace directory. "
    "Do NOT read or write files outside your workspace — those operations will fail."
)

# ---------------------------------------------------------------------------
# opencode binary setup (runs once at startup)
# ---------------------------------------------------------------------------

def _install_opencode():
    """Install / upgrade the opencode binary to the latest version."""
    print("Installing latest opencode binary...")
    subprocess.run(
        "curl -fsSL https://opencode.ai/install | bash",
        shell=True, timeout=120, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _ensure_opencode():
    """Install the opencode binary if not already present."""
    opencode_bin = os.path.join(os.path.expanduser("~"), ".opencode", "bin")
    if opencode_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = opencode_bin + os.pathsep + os.environ.get("PATH", "")
    if not shutil.which("opencode"):
        _install_opencode()
    if not shutil.which("opencode"):
        raise RuntimeError("opencode installation failed")


def _write_agent_config(agent_dir, model_name, port):
    """Write opencode.json for a specific model.

    Called before each server start (including retries with a different
    model).  Only the selected model is registered so opencode uses it.

    Args:
        agent_dir: Path to the agent's working directory.
        model_name: Display name from available_models (e.g. "OpenAI: GPT-5.2-Codex").
        port: TCP port for the opencode server.
    """
    model_id = model_name_to_id[model_name]
    context_window = model_context_window[model_name]
    display = model_id.split("/", 1)[-1] if "/" in model_id else model_id

    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "openrouter": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "OpenRouter",
                "options": {
                    "baseURL": "https://openrouter.ai/api/v1",
                    "apiKey": "{env:OPENROUTER_API_KEY}",
                },
                "models": {
                    model_id: {
                        "name": display,
                        "limit": {
                            "context": context_window,
                            "output": 65536,
                        },
                    },
                },
            },
        },
        "server": {
            "port": port,
        },
        "model": f"openrouter/{model_id}",
    }
    config_path = os.path.join(agent_dir, "opencode.json")
    with open(config_path, "w") as f:
        json.dump(config, f)


# ---------------------------------------------------------------------------
# opencode server management
# ---------------------------------------------------------------------------

# Global registry: port -> subprocess.Popen
_server_processes = {}


def find_free_port():
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_opencode_server(agent_dir, port):
    """Start opencode in headless server mode.

    Args:
        agent_dir: Working directory (must have opencode.json).
        port: TCP port to listen on.

    Returns:
        The port number.
    """
    # Isolate every opencode instance inside agent_dir so concurrent servers
    # never share state and stale files from a killed process don't bleed
    # into the next run.
    #
    # XDG_DATA_HOME — opencode's SQLite DB; per-instance avoids "locked" errors.
    # TMPDIR/TEMP/TMP — opencode writes internal temp files (incl. the JSONL
    #   output schema) to the system temp dir using the binary name as a key
    #   ("arg0 temp dirs").  Redirecting these into agent_dir means:
    #     1. No cross-instance collisions between concurrent battles.
    #     2. The stale-dir cleanup warning disappears because each dir is
    #        fresh (UUID-named) and removed by shutil.rmtree on teardown.
    #     3. The "Failed to read output schema file jsonl" error is gone
    #        because the schema is always written fresh into the new dir.
    # HOME — catches any ~/... path expansion that might escape the sandbox.
    xdg_data = os.path.join(agent_dir, ".xdg_data")
    agent_tmp = os.path.join(agent_dir, ".tmp")
    os.makedirs(xdg_data, exist_ok=True)
    os.makedirs(agent_tmp, exist_ok=True)
    env = os.environ.copy()
    env["XDG_DATA_HOME"] = xdg_data
    env["TMPDIR"] = agent_tmp
    env["TEMP"] = agent_tmp
    env["TMP"] = agent_tmp
    env["HOME"] = agent_dir

    proc = subprocess.Popen(
        ["opencode", "serve", "--port", str(port)],
        cwd=agent_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _server_processes[port] = proc
    _wait_for_server(port)
    return port


def _wait_for_server(port, timeout=30):
    """Poll until the opencode server is accepting connections."""
    deadline = time.time() + timeout
    url = f"http://localhost:{port}/global/health"
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code < 500:
                return
        except (httpx.ConnectError, httpx.ReadError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"opencode server on port {port} not ready after {timeout}s")


def stop_opencode_server(port):
    """Terminate an opencode server process."""
    proc = _server_processes.pop(port, None)
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# Initialize opencode binary
_ensure_opencode()


def _run_agent_in_thread(agent_dir, port, prompt, preferred_model, global_deadline):
    """Synchronous wrapper around run_agent_with_retry for use in threads.

    Each call spins up its own event loop so multiple threads can run
    async agent logic concurrently without sharing a loop.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            run_agent_with_retry(
                agent_dir, port, prompt,
                preferred_model=preferred_model,
                global_deadline=global_deadline,
            )
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Model metadata (loaded from individual JSON files in HF dataset repo)
# ---------------------------------------------------------------------------

# Load model metadata from Hugging Face
model_context_window = {}
model_name_to_id = {}
model_organization = {}
available_models = []
active_models = []  # Only models with state != "inactive" (used for pairwise selection)

_api = HfApi()
for _file in _api.list_repo_files(repo_id=MODEL_REPO, repo_type="dataset"):
    if not _file.endswith(".json"):
        continue
    _local_path = hf_hub_download(repo_id=MODEL_REPO, filename=_file, repo_type="dataset")
    with open(_local_path, "r") as f:
        _record = json.load(f)
    # model_name is derived from the filename (without .json extension)
    _model_name = _file.rsplit("/", 1)[-1].replace(".json", "")
    available_models.append(_model_name)
    model_context_window[_model_name] = _record["context_window"]
    model_name_to_id[_model_name] = _record["id"]
    model_organization[_model_name] = _model_name.split(": ")[0]
    # Track active models for pairwise selection (must be active and support tool calling)
    if _record.get("state") != "inactive" and _record.get("tool_calling") is True:
        active_models.append(_model_name)



# ---------------------------------------------------------------------------
# URL parsing helpers
# ---------------------------------------------------------------------------

def _parse_url_path(url):
    """Parse a URL and return (hostname, path_segments)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        segments = [s for s in parsed.path.split("/") if s]
        return hostname, segments
    except Exception:
        return None, []


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def _classify_github_url(segments):
    """Classify a GitHub URL from its path segments into resource type + params."""
    if len(segments) < 2:
        return None

    owner, repo = segments[0], segments[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    base = {"owner": owner, "repo": repo}

    if len(segments) == 2:
        return {**base, "resource": None}

    res = segments[2]

    if res == "issues" and len(segments) >= 4:
        return {**base, "resource": "issues", "id": segments[3]}
    elif res == "pull" and len(segments) >= 4:
        return {**base, "resource": "pull", "id": segments[3]}
    elif res == "commit" and len(segments) >= 4:
        return {**base, "resource": "commit", "sha": segments[3]}
    elif res == "blob" and len(segments) >= 4:
        return {**base, "resource": "blob", "branch": segments[3],
                "path": "/".join(segments[4:]) if len(segments) > 4 else ""}
    elif res == "tree" and len(segments) >= 4:
        return {**base, "resource": "tree", "branch": segments[3],
                "path": "/".join(segments[4:]) if len(segments) > 4 else ""}
    elif res == "discussions" and len(segments) >= 4:
        return {**base, "resource": "discussions", "id": segments[3]}
    elif res == "releases" and len(segments) >= 5 and segments[3] == "tag":
        return {**base, "resource": "releases", "tag": segments[4]}
    elif res == "compare" and len(segments) >= 4:
        return {**base, "resource": "compare", "spec": segments[3]}
    elif res == "actions" and len(segments) >= 5 and segments[3] == "runs":
        return {**base, "resource": "actions", "run_id": segments[4]}
    elif res == "wiki":
        page = segments[3] if len(segments) >= 4 else None
        return {**base, "resource": "wiki", "page": page}
    else:
        return {**base, "resource": "unknown"}


def _fmt_github_repo(repo):
    parts = [f"Repository: {repo.full_name}"]
    if repo.description:
        parts.append(f"Description: {repo.description}")
    try:
        readme = repo.get_readme()
        content = readme.decoded_content.decode("utf-8", errors="replace")
        parts.append(f"README (first 2000 chars):\n{content[:2000]}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_github_issue(repo, issue_id):
    issue = repo.get_issue(issue_id)
    parts = [
        f"Issue #{issue.number}: {issue.title}",
        f"State: {issue.state}",
        f"Body:\n{issue.body or '(empty)'}",
    ]
    comments = issue.get_comments()
    comment_texts = []
    for i, c in enumerate(comments):
        if i >= 10:
            break
        comment_texts.append(f"  Comment by {c.user.login}:\n  {c.body}")
    if comment_texts:
        parts.append("Comments (first 10):\n" + "\n---\n".join(comment_texts))
    return "\n\n".join(parts)


def _fmt_github_pr(repo, pr_id):
    pr = repo.get_pull(pr_id)
    parts = [
        f"Pull Request #{pr.number}: {pr.title}",
        f"State: {pr.state}  Merged: {pr.merged}",
        f"Body:\n{pr.body or '(empty)'}",
    ]
    diff_parts = []
    for f in pr.get_files():
        header = f"--- {f.filename} ({f.status}, +{f.additions}/-{f.deletions})"
        patch = f.patch or "(binary or too large)"
        diff_parts.append(f"{header}\n{patch}")
    if diff_parts:
        diff_text = "\n\n".join(diff_parts)
        if len(diff_text) > 5000:
            diff_text = diff_text[:5000] + "\n... (diff truncated)"
        parts.append(f"Diff:\n{diff_text}")
    return "\n\n".join(parts)


def _fmt_github_commit(repo, sha):
    commit = repo.get_commit(sha)
    parts = [
        f"Commit: {commit.sha}",
        f"Message: {commit.commit.message}",
        f"Author: {commit.commit.author.name}",
        f"Stats: +{commit.stats.additions}/-{commit.stats.deletions}",
    ]
    file_patches = []
    for f in commit.files:
        file_patches.append(f"  {f.filename} ({f.status}): {f.patch or '(binary)'}")
    if file_patches:
        patch_text = "\n".join(file_patches)
        if len(patch_text) > 5000:
            patch_text = patch_text[:5000] + "\n... (patch truncated)"
        parts.append(f"Files changed:\n{patch_text}")
    return "\n\n".join(parts)


def _fmt_github_blob(repo, branch, path):
    contents = repo.get_contents(path, ref=branch)
    if isinstance(contents, list):
        listing = "\n".join(f"  {c.path} ({c.type})" for c in contents)
        return f"Directory listing at {branch}/{path}:\n{listing}"
    content = contents.decoded_content.decode("utf-8", errors="replace")
    if len(content) > 5000:
        content = content[:5000] + "\n... (content truncated)"
    return f"File: {path} (branch: {branch})\n\n{content}"


def _fmt_github_tree(repo, branch, path):
    if path:
        contents = repo.get_contents(path, ref=branch)
        if not isinstance(contents, list):
            contents = [contents]
    else:
        contents = repo.get_contents("", ref=branch)
    listing = "\n".join(f"  {c.path} ({c.type}, {c.size} bytes)" for c in contents)
    return f"Tree at {branch}/{path or '(root)'}:\n{listing}"


_DISCUSSION_GRAPHQL_SCHEMA = """
    title
    body
    number
    author { login }
    comments(first: 10) {
        nodes {
            body
            author { login }
        }
    }
"""


def _fmt_github_discussion(repo, discussion_id):
    try:
        discussion = repo.get_discussion(discussion_id, _DISCUSSION_GRAPHQL_SCHEMA)
        parts = [
            f"Discussion #{discussion.number}: {discussion.title}",
            f"Body:\n{discussion.body or '(empty)'}",
        ]
        if hasattr(discussion, "comments") and discussion.comments:
            comment_texts = []
            for c in discussion.comments:
                author = c.author.login if hasattr(c, "author") and c.author else "unknown"
                comment_texts.append(f"  Comment by {author}: {c.body}")
            if comment_texts:
                parts.append("Comments:\n" + "\n---\n".join(comment_texts))
        return "\n\n".join(parts)
    except Exception as e:
        print(f"Discussion fetch failed (GraphQL): {e}")
        return None


def _fmt_github_release(repo, tag):
    release = repo.get_release(tag)
    parts = [
        f"Release: {release.title or release.tag_name}",
        f"Tag: {release.tag_name}",
        f"Body:\n{release.body or '(empty)'}",
    ]
    return "\n\n".join(parts)


def _fmt_github_compare(repo, spec):
    if "..." in spec:
        base, head = spec.split("...", 1)
    elif ".." in spec:
        base, head = spec.split("..", 1)
    else:
        return None
    comparison = repo.compare(base, head)
    parts = [
        f"Comparison: {base}...{head}",
        f"Status: {comparison.status}",
        f"Ahead by: {comparison.ahead_by}, Behind by: {comparison.behind_by}",
        f"Total commits: {comparison.total_commits}",
    ]
    commit_summaries = []
    for c in comparison.commits[:20]:
        commit_summaries.append(f"  {c.sha[:8]}: {c.commit.message.splitlines()[0]}")
    if commit_summaries:
        parts.append("Commits:\n" + "\n".join(commit_summaries))
    file_summaries = []
    for f in comparison.files[:30]:
        file_summaries.append(f"  {f.filename} ({f.status}, +{f.additions}/-{f.deletions})")
    if file_summaries:
        parts.append("Files changed:\n" + "\n".join(file_summaries))
    return "\n\n".join(parts)


def _fmt_github_actions(repo, run_id):
    run = repo.get_workflow_run(run_id)
    parts = [
        f"Workflow Run: {run.name} #{run.run_number}",
        f"Status: {run.status}  Conclusion: {run.conclusion}",
        f"SHA: {run.head_sha}",
    ]
    try:
        jobs = run.jobs()
        for job in jobs:
            if job.conclusion == "failure":
                parts.append(f"Failed job: {job.name}")
                for step in job.steps:
                    if step.conclusion == "failure":
                        parts.append(f"  Failed step: {step.name}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_github_wiki(owner, repo_name, page):
    if page:
        return f"Wiki page: {page} (from {owner}/{repo_name}/wiki)\nNote: Wiki content cannot be fetched via API."
    return f"Wiki: {owner}/{repo_name}/wiki\nNote: Wiki content cannot be fetched via API."


def fetch_github_content(url):
    """Fetch detailed content from a GitHub URL using PyGithub."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set.")
        return None

    g = Github(auth=Auth.Token(token))
    hostname, segments = _parse_url_path(url)

    if not hostname or "github.com" not in hostname:
        return None

    info = _classify_github_url(segments)
    if not info:
        return None

    try:
        repo = g.get_repo(f"{info['owner']}/{info['repo']}")
        resource = info["resource"]

        if resource is None:
            return _fmt_github_repo(repo)
        elif resource == "issues":
            return _fmt_github_issue(repo, int(info["id"]))
        elif resource == "pull":
            return _fmt_github_pr(repo, int(info["id"]))
        elif resource == "commit":
            return _fmt_github_commit(repo, info["sha"])
        elif resource == "blob":
            return _fmt_github_blob(repo, info["branch"], info["path"])
        elif resource == "tree":
            return _fmt_github_tree(repo, info["branch"], info.get("path", ""))
        elif resource == "discussions":
            return _fmt_github_discussion(repo, int(info["id"]))
        elif resource == "releases":
            return _fmt_github_release(repo, info["tag"])
        elif resource == "compare":
            return _fmt_github_compare(repo, info["spec"])
        elif resource == "actions":
            return _fmt_github_actions(repo, int(info["run_id"]))
        elif resource == "wiki":
            return _fmt_github_wiki(info["owner"], info["repo"], info.get("page"))
        else:
            return None
    except Exception as e:
        print(f"GitHub API error: {e}")
        return None


# ---------------------------------------------------------------------------
# GitLab
# ---------------------------------------------------------------------------

def _classify_gitlab_url(segments):
    """Classify a GitLab URL from its path segments."""
    try:
        dash_idx = segments.index("-")
    except ValueError:
        if len(segments) >= 2:
            return {"project_path": "/".join(segments), "resource": None}
        return None

    project_path = "/".join(segments[:dash_idx])
    res_segments = segments[dash_idx + 1:]

    if not project_path or not res_segments:
        return {"project_path": project_path, "resource": None}

    res = res_segments[0]

    if res == "issues" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "issues", "id": res_segments[1]}
    elif res == "merge_requests" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "merge_requests", "id": res_segments[1]}
    elif res in ("commit", "commits") and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "commit", "sha": res_segments[1]}
    elif res == "blob" and len(res_segments) >= 2:
        branch = res_segments[1]
        file_path = "/".join(res_segments[2:]) if len(res_segments) > 2 else ""
        return {"project_path": project_path, "resource": "blob", "branch": branch, "path": file_path}
    elif res == "tree" and len(res_segments) >= 2:
        branch = res_segments[1]
        tree_path = "/".join(res_segments[2:]) if len(res_segments) > 2 else ""
        return {"project_path": project_path, "resource": "tree", "branch": branch, "path": tree_path}
    elif res == "releases" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "releases", "tag": res_segments[1]}
    elif res == "compare" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "compare", "spec": res_segments[1]}
    elif res == "pipelines" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "pipelines", "id": res_segments[1]}
    elif res == "wikis":
        page = res_segments[1] if len(res_segments) >= 2 else None
        return {"project_path": project_path, "resource": "wikis", "page": page}
    else:
        return {"project_path": project_path, "resource": "unknown"}


def _fmt_gitlab_repo(project):
    parts = [f"Repository: {project.path_with_namespace}"]
    if project.description:
        parts.append(f"Description: {project.description}")
    try:
        readme = project.files.get(file_path="README.md", ref=project.default_branch)
        content = readme.decode().decode("utf-8", errors="replace")
        parts.append(f"README (first 2000 chars):\n{content[:2000]}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_gitlab_issue(project, issue_id):
    issue = project.issues.get(issue_id)
    parts = [
        f"Issue #{issue.iid}: {issue.title}",
        f"State: {issue.state}",
        f"Body:\n{issue.description or '(empty)'}",
    ]
    notes = issue.notes.list(get_all=False, per_page=10)
    note_texts = [f"  Comment by {n.author['username']}: {n.body}" for n in notes]
    if note_texts:
        parts.append("Comments (first 10):\n" + "\n---\n".join(note_texts))
    return "\n\n".join(parts)


def _fmt_gitlab_mr(project, mr_id):
    mr = project.mergerequests.get(mr_id)
    parts = [
        f"Merge Request !{mr.iid}: {mr.title}",
        f"State: {mr.state}",
        f"Body:\n{mr.description or '(empty)'}",
    ]
    try:
        changes = mr.changes()
        if isinstance(changes, dict) and "changes" in changes:
            diff_parts = []
            for change in changes["changes"][:30]:
                diff_parts.append(f"  {change.get('new_path', '?')}: {change.get('diff', '')[:500]}")
            if diff_parts:
                diff_text = "\n".join(diff_parts)
                if len(diff_text) > 5000:
                    diff_text = diff_text[:5000] + "\n... (diff truncated)"
                parts.append(f"Changes:\n{diff_text}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_gitlab_commit(project, sha):
    commit = project.commits.get(sha)
    parts = [
        f"Commit: {commit.id}",
        f"Title: {commit.title}",
        f"Message: {commit.message}",
        f"Author: {commit.author_name}",
    ]
    try:
        diffs = commit.diff()
        diff_parts = []
        for d in diffs[:30]:
            diff_parts.append(f"  {d.get('new_path', '?')}: {d.get('diff', '')[:500]}")
        if diff_parts:
            diff_text = "\n".join(diff_parts)
            if len(diff_text) > 5000:
                diff_text = diff_text[:5000] + "\n... (diff truncated)"
            parts.append(f"Diff:\n{diff_text}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_gitlab_blob(project, branch, path):
    f = project.files.get(file_path=path, ref=branch)
    content = f.decode().decode("utf-8", errors="replace")
    if len(content) > 5000:
        content = content[:5000] + "\n... (content truncated)"
    return f"File: {path} (branch: {branch})\n\n{content}"


def _fmt_gitlab_tree(project, branch, path):
    items = project.repository_tree(path=path or "", ref=branch, get_all=False, per_page=100)
    listing = "\n".join(f"  {item['path']} ({item['type']})" for item in items)
    return f"Tree at {branch}/{path or '(root)'}:\n{listing}"


def _fmt_gitlab_release(project, tag):
    release = project.releases.get(tag)
    parts = [
        f"Release: {release.name or release.tag_name}",
        f"Tag: {release.tag_name}",
        f"Description:\n{release.description or '(empty)'}",
    ]
    return "\n\n".join(parts)


def _fmt_gitlab_compare(project, spec):
    if "..." in spec:
        base, head = spec.split("...", 1)
    elif ".." in spec:
        base, head = spec.split("..", 1)
    else:
        return None
    result = project.repository_compare(base, head)
    parts = [f"Comparison: {base}...{head}"]
    if isinstance(result, dict):
        commits = result.get("commits", [])
        commit_summaries = []
        for c in commits[:20]:
            commit_summaries.append(f"  {c.get('short_id', '?')}: {c.get('title', '')}")
        if commit_summaries:
            parts.append("Commits:\n" + "\n".join(commit_summaries))
        diffs = result.get("diffs", [])
        diff_parts = []
        for d in diffs[:30]:
            diff_parts.append(f"  {d.get('new_path', '?')}: {d.get('diff', '')[:500]}")
        if diff_parts:
            diff_text = "\n".join(diff_parts)
            if len(diff_text) > 5000:
                diff_text = diff_text[:5000] + "\n... (diff truncated)"
            parts.append(f"Diffs:\n{diff_text}")
    return "\n\n".join(parts)


def _fmt_gitlab_pipeline(project, pipeline_id):
    pipeline = project.pipelines.get(pipeline_id)
    parts = [
        f"Pipeline #{pipeline.id}",
        f"Status: {pipeline.status}",
        f"Ref: {pipeline.ref}",
        f"SHA: {pipeline.sha}",
    ]
    try:
        jobs = pipeline.jobs.list(get_all=False, per_page=20)
        failed_jobs = [j for j in jobs if j.status == "failed"]
        if failed_jobs:
            parts.append("Failed jobs:")
            for j in failed_jobs:
                parts.append(f"  {j.name}: {j.status} (stage: {j.stage})")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_gitlab_wiki(project, page):
    if page:
        try:
            wiki_page = project.wikis.get(page)
            return f"Wiki page: {wiki_page.title}\n\n{wiki_page.content}"
        except Exception:
            return f"Wiki page: {page}\nNote: Could not fetch wiki page content."
    try:
        pages = project.wikis.list(get_all=False, per_page=20)
        listing = "\n".join(f"  {p.slug}: {p.title}" for p in pages)
        return f"Wiki pages:\n{listing}"
    except Exception:
        return "Wiki: Could not fetch wiki pages."


def fetch_gitlab_content(url):
    """Fetch content from GitLab URL using python-gitlab."""
    token = os.getenv("GITLAB_TOKEN")
    if not token:
        print("GITLAB_TOKEN not set.")
        return None

    gl = gitlab.Gitlab("https://gitlab.com", private_token=token)
    hostname, segments = _parse_url_path(url)

    if not hostname or "gitlab.com" not in hostname:
        return None

    info = _classify_gitlab_url(segments)
    if not info:
        return None

    try:
        project = gl.projects.get(info["project_path"])
        resource = info["resource"]

        if resource is None:
            return _fmt_gitlab_repo(project)
        elif resource == "issues":
            return _fmt_gitlab_issue(project, int(info["id"]))
        elif resource == "merge_requests":
            return _fmt_gitlab_mr(project, int(info["id"]))
        elif resource == "commit":
            return _fmt_gitlab_commit(project, info["sha"])
        elif resource == "blob":
            return _fmt_gitlab_blob(project, info["branch"], info["path"])
        elif resource == "tree":
            return _fmt_gitlab_tree(project, info["branch"], info.get("path", ""))
        elif resource == "releases":
            return _fmt_gitlab_release(project, info["tag"])
        elif resource == "compare":
            return _fmt_gitlab_compare(project, info["spec"])
        elif resource == "pipelines":
            return _fmt_gitlab_pipeline(project, int(info["id"]))
        elif resource == "wikis":
            return _fmt_gitlab_wiki(project, info.get("page"))
        else:
            return None
    except Exception as e:
        print(f"GitLab API error: {e}")
        return None


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

def _classify_huggingface_url(segments):
    """Classify a HuggingFace URL from its path segments."""
    if not segments:
        return None

    repo_type = None
    segs = list(segments)
    if segs[0] in ("datasets", "spaces"):
        repo_type = segs[0].rstrip("s")
        segs = segs[1:]

    if len(segs) < 2:
        return None

    repo_id = f"{segs[0]}/{segs[1]}"
    base = {"repo_id": repo_id, "repo_type": repo_type}

    if len(segs) == 2:
        return {**base, "resource": None}

    res = segs[2]

    if res == "blob" and len(segs) >= 4:
        return {**base, "resource": "blob", "revision": segs[3],
                "path": "/".join(segs[4:]) if len(segs) > 4 else ""}
    elif res == "resolve" and len(segs) >= 4:
        return {**base, "resource": "resolve", "revision": segs[3],
                "path": "/".join(segs[4:]) if len(segs) > 4 else ""}
    elif res == "tree" and len(segs) >= 4:
        return {**base, "resource": "tree", "revision": segs[3],
                "path": "/".join(segs[4:]) if len(segs) > 4 else ""}
    elif res == "commit" and len(segs) >= 4:
        return {**base, "resource": "commit", "sha": segs[3]}
    elif res == "discussions" and len(segs) >= 4:
        return {**base, "resource": "discussions", "num": segs[3]}
    else:
        return {**base, "resource": "unknown"}


def _fmt_hf_repo(api, repo_id, repo_type):
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    parts = [f"Repository: {repo_id}"]
    if hasattr(info, "description") and info.description:
        parts.append(f"Description: {info.description}")
    if hasattr(info, "card_data") and info.card_data:
        parts.append(f"Card data: {str(info.card_data)[:1000]}")
    try:
        readme_path = api.hf_hub_download(
            repo_id=repo_id, filename="README.md", repo_type=repo_type
        )
        with open(readme_path, "r", errors="replace") as f:
            content = f.read()[:2000]
        parts.append(f"README (first 2000 chars):\n{content}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_hf_commit(api, repo_id, repo_type, sha):
    commits = api.list_repo_commits(repo_id=repo_id, revision=sha, repo_type=repo_type)
    if commits:
        c = commits[0]
        return (
            f"Commit: {c.commit_id}\n"
            f"Title: {c.title}\n"
            f"Message: {c.message}\n"
            f"Authors: {', '.join(c.authors) if c.authors else 'unknown'}\n"
            f"Date: {c.created_at}"
        )
    return None


def _fmt_hf_discussion(api, repo_id, repo_type, discussion_num):
    discussion = api.get_discussion_details(
        repo_id=repo_id, discussion_num=discussion_num, repo_type=repo_type
    )
    parts = [
        f"Discussion #{discussion.num}: {discussion.title}",
        f"Status: {discussion.status}",
        f"Author: {discussion.author}",
        f"Is Pull Request: {discussion.is_pull_request}",
    ]
    comment_texts = []
    for event in discussion.events:
        if hasattr(event, "content") and event.content:
            author = event.author if hasattr(event, "author") else "unknown"
            comment_texts.append(f"  {author}: {event.content[:500]}")
        if len(comment_texts) >= 10:
            break
    if comment_texts:
        parts.append("Comments:\n" + "\n---\n".join(comment_texts))
    return "\n\n".join(parts)


def _fmt_hf_file(api, repo_id, repo_type, revision, path):
    local_path = api.hf_hub_download(
        repo_id=repo_id, filename=path, revision=revision, repo_type=repo_type
    )
    try:
        with open(local_path, "r", errors="replace") as f:
            content = f.read()
        if len(content) > 5000:
            content = content[:5000] + "\n... (content truncated)"
        return f"File: {path} (revision: {revision})\n\n{content}"
    except Exception:
        return f"File: {path} (revision: {revision})\n(binary or unreadable file)"


def _fmt_hf_tree(api, repo_id, repo_type, revision, path):
    items = api.list_repo_tree(
        repo_id=repo_id, path_in_repo=path or None,
        revision=revision, repo_type=repo_type
    )
    listing = []
    for item in items:
        if hasattr(item, "size") and item.size is not None:
            listing.append(f"  {item.rfilename} (file, {item.size} bytes)")
        else:
            listing.append(f"  {item.rfilename} (folder)")
        if len(listing) >= 100:
            listing.append("  ... (truncated)")
            break
    return f"Tree at {revision}/{path or '(root)'}:\n" + "\n".join(listing)


def fetch_huggingface_content(url):
    """Fetch detailed content from a Hugging Face URL using huggingface_hub API."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set.")
        return None

    api = HfApi(token=token)
    hostname, segments = _parse_url_path(url)

    if not hostname or "huggingface.co" not in hostname:
        return None

    info = _classify_huggingface_url(segments)
    if not info:
        return None

    try:
        resource = info["resource"]
        repo_id = info["repo_id"]
        repo_type = info["repo_type"]

        if resource is None:
            return _fmt_hf_repo(api, repo_id, repo_type)
        elif resource == "commit":
            return _fmt_hf_commit(api, repo_id, repo_type, info["sha"])
        elif resource == "discussions":
            return _fmt_hf_discussion(api, repo_id, repo_type, int(info["num"]))
        elif resource in ("blob", "resolve"):
            return _fmt_hf_file(api, repo_id, repo_type, info["revision"], info["path"])
        elif resource == "tree":
            return _fmt_hf_tree(api, repo_id, repo_type, info["revision"], info.get("path", ""))
        else:
            return None
    except Exception as e:
        print(f"Hugging Face API error: {e}")
        return None


# ---------------------------------------------------------------------------
# URL router
# ---------------------------------------------------------------------------

def fetch_url_content(url):
    """Main URL content fetcher that routes to platform-specific handlers."""
    if not url or not url.strip():
        return ""
    url = url.strip()
    try:
        hostname, _ = _parse_url_path(url)
        if hostname and "github.com" in hostname:
            return fetch_github_content(url)
        elif hostname and "gitlab.com" in hostname:
            return fetch_gitlab_content(url)
        elif hostname and "huggingface.co" in hostname:
            return fetch_huggingface_content(url)
    except Exception as e:
        print(f"Error fetching URL content: {e}")
    return ""


# ---------------------------------------------------------------------------
# Folder validation helpers
# ---------------------------------------------------------------------------

def detect_folder_violation_error(error_message, agent_dir):
    """Detect if an error indicates the agent tried to access files outside its directory.
    
    Args:
        error_message: The error message from the agent
        agent_dir: The designated working directory for the agent
    
    Returns:
        bool: True if this appears to be a folder violation error
    """
    if not error_message:
        return False
    
    error_str = str(error_message).lower()
    
    # Common patterns indicating folder violations
    violation_patterns = [
        "permission denied",
        "no such file or directory",
        "cannot access", 
        "operation not permitted",
        "access denied",
        "file not found",
        "path not found",
        "directory not found"
    ]
    
    # Check if error contains violation patterns AND references paths outside agent_dir
    has_violation_pattern = any(pattern in error_str for pattern in violation_patterns)
    
    if has_violation_pattern:
        # Look for absolute path references that are outside the agent directory
        absolute_paths = re.findall(r'[/\\][a-zA-Z0-9_/\\.-]+', error_message)
        for path in absolute_paths:
            if agent_dir not in path and not path.startswith('./') and not path.startswith('../'):
                return True
                
        # Look for common problematic paths
        problematic_paths = [
            '/tmp/', '/home/', '/usr/', '/var/', '/etc/', '/opt/', '/root/',
            'c:\\', 'd:\\', 'c:/', 'd:/', '~/'
        ]
        if any(bad_path in error_str for bad_path in problematic_paths):
            return True
    
    return False


def analyze_agent_output_for_violations(output, error, agent_dir):
    """Analyze agent output and errors for folder violations.
    
    Args:
        output: Agent's text output
        error: Agent's error message (if any)
        agent_dir: The designated working directory
    
    Returns:
        bool: True if folder violations detected
    """
    # Check explicit error messages
    if error and detect_folder_violation_error(error, agent_dir):
        return True
    
    # Check output for violation indicators
    if output:
        output_str = str(output).lower()
        violation_indicators = [
            "cannot create",
            "cannot write", 
            "cannot read",
            "access denied",
            "permission denied",
            "file not found",
            "no such file"
        ]
        
        # Also check for absolute path usage in output
        if any(indicator in output_str for indicator in violation_indicators):
            absolute_paths = re.findall(r'[/\\][a-zA-Z0-9_/\\.-]+', output)
            for path in absolute_paths:
                if agent_dir not in path:
                    return True
    
    return False


# ---------------------------------------------------------------------------
# opencode agent dispatcher (SDK-based with session continuity)
# ---------------------------------------------------------------------------

def extract_output(messages):
    """Extract readable text from opencode SDK ``SessionMessagesResponse``.

    Iterates over the message list returned by ``client.session.messages()``,
    filters to assistant-role messages, and collects text parts and completed
    tool parts.  Other part types (step_start, step_finish, snapshot, patch)
    are silently skipped.

    Args:
        messages: ``SessionMessagesResponse`` — a list of
            ``SessionMessagesResponseItem`` objects, each with ``.info``
            and ``.parts``.
    """
    parts_list = []
    for msg in messages:
        # Only extract from assistant messages
        if getattr(msg.info, "role", None) != "assistant":
            continue

        for part in msg.parts:
            ptype = getattr(part, "type", None)
            if ptype == "text":
                text = getattr(part, "text", "")
                if text:
                    parts_list.append(text)
            elif ptype == "tool":
                tool_name = getattr(part, "tool", "unknown")
                state = getattr(part, "state", None)
                if state is None:
                    continue
                status = getattr(state, "status", "")
                title = getattr(state, "title", "")
                if status == "completed":
                    output = getattr(state, "output", "")
                    label = f"[Tool: {tool_name}]"
                    if title:
                        label += f" {title}"
                    if output:
                        parts_list.append(f"{label}\n{output}")
                    else:
                        parts_list.append(label)
                elif status == "error":
                    error = getattr(state, "error", "unknown error")
                    parts_list.append(f"[Tool: {tool_name}] Error: {error}")
    return "\n\n".join(parts_list)


async def run_agent(port, model_id, prompt, session_id=None):
    """Run a single opencode agent invocation via the Python SDK.

    Uses ``AsyncOpencode`` to create a session, send the prompt, and
    poll for completion.  ``session.chat()`` is non-blocking — it kicks
    off the agent and returns immediately.  We poll
    ``session.messages()`` until the assistant message's
    ``time.completed`` is set (agent finished) or we timeout.

    Args:
        port: The opencode server port for this agent.
        model_id: OpenRouter model ID (e.g. "openai/gpt-5.2-codex").
        prompt: The user prompt (with optional repo context prepended).
        session_id: If provided, resume this session (follow-up round).

    Returns:
        dict with keys: ok, output, session_id, error (if failed)
    """
    base_url = f"http://localhost:{port}"
    try:
        async with AsyncOpencode(
            base_url=base_url,
            timeout=httpx.Timeout(AGENT_TIMEOUT, connect=30),
        ) as client:
            # Create session if needed
            if session_id is None:
                # extra_body={} ensures the SDK sends '{}' instead of
                # 'null' which the opencode server rejects as malformed.
                session = await client.session.create(extra_body={})
                session_id = session.id
                print(f"[Agent:{port}] Created session: {session_id}")

            # Send message — kicks off the agent (non-blocking)
            print(f"[Agent:{port}] Sending message (model={model_id})...")
            try:
                assistant_msg = await client.session.chat(
                    id=session_id,
                    model_id=model_id,
                    provider_id="openrouter",
                    parts=[{"type": "text", "text": prompt}],
                )
            except Exception as chat_err:
                # Log the full error details for debugging
                if hasattr(chat_err, "response"):
                    try:
                        body = chat_err.response.content[:500].decode("utf-8", errors="replace")
                    except Exception:
                        body = "(unreadable)"
                    print(f"[Agent:{port}] chat() error response: "
                          f"status={chat_err.response.status_code} "
                          f"body={body}")
                if hasattr(chat_err, "request"):
                    req = chat_err.request
                    print(f"[Agent:{port}] chat() request: "
                          f"method={req.method} url={req.url} "
                          f"body={req.content[:500] if req.content else 'empty'}")
                raise
            print(f"[Agent:{port}] chat() returned, polling for completion...")

            # ----------------------------------------------------------
            # Poll until the agent completes.  The assistant message's
            # time.completed transitions from None -> timestamp when the
            # agentic loop finishes.
            # ----------------------------------------------------------
            poll_interval = 3  # seconds between polls
            deadline = time.time() + AGENT_TIMEOUT
            messages = []

            while time.time() < deadline:
                await asyncio.sleep(poll_interval)
                try:
                    messages = await client.session.messages(session_id)
                except UnicodeDecodeError:
                    # The opencode server may include binary file content
                    # in session messages, causing UTF-8 decode failures.
                    # Skip this poll and retry on the next iteration.
                    print(f"[Agent:{port}] Skipping poll — response contained non-UTF-8 data")
                    continue

                # Find the last assistant message and check completion
                for msg in reversed(messages):
                    info = msg.info
                    if getattr(info, "role", None) != "assistant":
                        continue

                    completed = getattr(getattr(info, "time", None), "completed", None)
                    error = getattr(info, "error", None)

                    if error:
                        error_name = getattr(error, "name", "unknown")
                        error_data = getattr(error, "data", None)
                        print(f"[Agent:{port}] Agent error: {error_name} data={error_data}")

                        # Detect retryable "model doesn't support tool use"
                        error_str = str(error_data) if error_data else ""
                        if "tool use" in error_str.lower() or "No endpoints found" in error_str:
                            print(f"[Agent:{port}] Model lacks tool-use support (retryable)")
                            return {
                                "ok": False, "output": "", "error": error_str,
                                "session_id": session_id, "retryable": True,
                            }

                        output = extract_output(messages)
                        if not output:
                            output = f"Model error: {error_name}"
                        return {"ok": True, "output": output, "session_id": session_id}

                    if completed is not None:
                        print(f"[Agent:{port}] Agent completed")
                        output = extract_output(messages)
                        return {"ok": True, "output": output, "session_id": session_id}

                    # Still running
                    parts_count = len(msg.parts)
                    print(f"[Agent:{port}] Running... (parts so far: {parts_count})")
                    break  # found assistant msg, not done yet

            # Timeout — abort the agent and return whatever we have
            print(f"[Agent:{port}] Timed out after {AGENT_TIMEOUT}s, aborting...")
            try:
                await client.session.abort(session_id)
            except Exception:
                pass
            output = extract_output(messages)
            if output:
                return {"ok": True, "output": output, "session_id": session_id}
            return {"ok": False, "output": "", "error": "Model timed out", "session_id": session_id}

    except Exception as e:
        # Detailed error logging for SDK exceptions
        error_detail = str(e)
        if hasattr(e, "status_code"):
            error_detail = f"HTTP {e.status_code}: {e}"
        if hasattr(e, "response") and e.response is not None:
            try:
                body_preview = e.response.content[:1000].decode("utf-8", errors="replace")
                print(f"[Agent:{port}] Error response body: {body_preview}")
            except Exception:
                pass
        if hasattr(e, "request") and e.request is not None:
            try:
                req = e.request
                req_body = req.content[:500] if req.content else b"(empty)"
                print(f"[Agent:{port}] Error request: {req.method} {req.url} "
                      f"body={req_body}")
            except Exception:
                pass
        print(f"[Agent:{port}] Error: {error_detail}")
        return {"ok": False, "output": "", "error": error_detail, "session_id": session_id}


async def run_agent_with_retry(agent_dir, port, prompt, preferred_model=None,
                               exclude_models=None, global_deadline=None):
    """Pick a model, configure + start opencode, run the agent.

    On a retryable error (model lacks tool-use support or is unavailable),
    stops the server, rewrites ``opencode.json`` with a different model,
    restarts, and tries again.  Respects ``global_deadline`` — if the
    total time budget is exhausted, returns whatever is available.

    Returns:
        (model_name, result_dict)
    """
    tried = set(exclude_models or [])
    model_name = None
    attempt = 0
    use_preferred = (
        preferred_model is not None and preferred_model not in tried
    )

    while True:
        # Check global deadline
        if global_deadline and time.time() >= global_deadline:
            print(f"[Agent:{port}] Global timeout reached, giving up")
            return model_name, {
                "ok": False, "output": "",
                "error": "Battle timeout — no model completed in time",
                "session_id": None,
            }

        candidates = [m for m in active_models if m not in tried]
        if not candidates:
            return model_name, {
                "ok": False, "output": "",
                "error": "Every available model was tried — none worked",
                "session_id": None,
            }

        if use_preferred:
            model_name = preferred_model
            use_preferred = False
        else:
            model_name = random.choice(candidates)

        model_id = model_name_to_id[model_name]
        attempt += 1

        # (Re)write config for this model and (re)start the server
        _write_agent_config(agent_dir, model_name, port)
        try:
            start_opencode_server(agent_dir, port)
        except Exception as e:
            print(f"[Agent:{port}] Server start failed for {model_name}: {e}")
            tried.add(model_name)
            continue

        print(f"[Agent:{port}] Attempt {attempt}/{len(available_models)}: model={model_name}")
        result = await run_agent(port, model_id, prompt)

        if result.get("ok"):
            # Check for folder violations in first round (this function is only called for first round)
            output = result.get("output", "")
            if analyze_agent_output_for_violations(output, None, agent_dir):
                print(f"[Agent:{port}] Model {model_name} violated folder constraints, retrying with another...")
                tried.add(model_name)
                stop_opencode_server(port)
                continue
            
            # Success — server stays running for follow-up rounds
            return model_name, result

        # Check if this was a retryable error or folder violation
        error_msg = result.get('error', 'unknown')
        output = result.get('output', '')
        
        # For first round, check if it's a folder violation
        if analyze_agent_output_for_violations(output, error_msg, agent_dir):
            print(f"[Agent:{port}] Model {model_name} violated folder constraints (error={error_msg}), retrying with another...")
        else:
            print(f"[Agent:{port}] Model {model_name} failed (error={error_msg}), retrying with another...")
        
        tried.add(model_name)
        stop_opencode_server(port)


async def run_first_round_with_retry(
    left_dir, right_dir, left_port, right_port, left_prompt, right_prompt
):
    """Run both agents in parallel, each with independent model retry.

    Pre-picks two *different* models so the left and right sides start
    with distinct models.  Each side retries independently (rewriting
    config + restarting server) if its model is not usable.  Both sides
    share a global deadline (``BATTLE_TIMEOUT``).

    ``left_prompt`` and ``right_prompt`` are built with the respective
    agent_dir already injected, so each agent knows its exact workspace.
    """
    global_deadline = time.time() + BATTLE_TIMEOUT

    left_preferred = random.choice(active_models)
    right_candidates = [m for m in active_models if m != left_preferred]
    right_preferred = random.choice(right_candidates) if right_candidates else left_preferred

    (left_name, result_a), (right_name, result_b) = await asyncio.gather(
        run_agent_with_retry(
            left_dir, left_port, left_prompt,
            preferred_model=left_preferred, global_deadline=global_deadline,
        ),
        run_agent_with_retry(
            right_dir, right_port, right_prompt,
            preferred_model=right_preferred, global_deadline=global_deadline,
        ),
    )
    return left_name, right_name, result_a, result_b


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(user_prompt, repo_context="", agent_dir=None):
    """Build the full prompt with system prefix and optional repo context.

    Args:
        user_prompt: The user's task description.
        repo_context: Optional fetched content from a repo URL.
        agent_dir: Absolute path to this agent's isolated workspace directory.
            When provided, it is injected into the prompt so the agent knows
            exactly where it is allowed to operate.
    """
    parts = [SYSTEM_PREFIX]
    if agent_dir:
        parts.append(
            f"Your workspace directory is: {agent_dir}\n"
            "All file operations must stay within this directory. "
            "You may use relative paths (they resolve here automatically) "
            "or absolute paths that start with this directory."
        )
    if repo_context:
        parts.append(f"Repository context:\n{repo_context}")
    parts.append(f"Inquiry: {user_prompt}")
    return "\n\n".join(parts)


def strip_context(prompt):
    """Remove the SYSTEM_PREFIX and repo context, returning just the user query."""
    marker = "\n\nInquiry: "
    idx = prompt.find(marker)
    return prompt[idx + len(marker):] if idx >= 0 else prompt


# ---------------------------------------------------------------------------
# Git operations (clone, checkout, diff)
# ---------------------------------------------------------------------------

def clone_repo(url, agent_dir):
    """Clone repository into agent_dir and checkout appropriate ref."""
    hostname, segments = _parse_url_path(url)
    if not hostname:
        return False

    parsed_info = None
    clone_url = None

    if "github.com" in hostname:
        parsed_info = _classify_github_url(segments)
        if not parsed_info:
            return False
        clone_url = f"https://github.com/{parsed_info['owner']}/{parsed_info['repo']}.git"
    elif "gitlab.com" in hostname:
        parsed_info = _classify_gitlab_url(segments)
        if not parsed_info:
            return False
        clone_url = f"https://gitlab.com/{parsed_info['project_path']}.git"
    elif "huggingface.co" in hostname:
        parsed_info = _classify_huggingface_url(segments)
        if not parsed_info:
            return False
        prefix = f"{parsed_info['repo_type']}s/" if parsed_info.get("repo_type") else ""
        clone_url = f"https://huggingface.co/{prefix}{parsed_info['repo_id']}"
    else:
        return False

    try:
        subprocess.run(
            ["git", "clone", "--depth=1", clone_url, "."],
            cwd=agent_dir, timeout=120, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _checkout_ref(parsed_info, agent_dir)
        return True
    except Exception:
        return False


def _checkout_ref(parsed_info, agent_dir):
    """Checkout specific ref after clone based on URL resource type."""
    resource = parsed_info.get("resource")
    try:
        if resource == "pull" and "id" in parsed_info:
            subprocess.run(
                ["git", "fetch", "origin", f"pull/{parsed_info['id']}/head:pr"],
                cwd=agent_dir, timeout=60, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "checkout", "pr"],
                cwd=agent_dir, timeout=30, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        elif resource == "merge_requests" and "id" in parsed_info:
            subprocess.run(
                ["git", "fetch", "origin", f"merge-requests/{parsed_info['id']}/head:mr"],
                cwd=agent_dir, timeout=60, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "checkout", "mr"],
                cwd=agent_dir, timeout=30, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        elif resource == "commit" and "sha" in parsed_info:
            subprocess.run(
                ["git", "fetch", "--depth=1", "origin", parsed_info["sha"]],
                cwd=agent_dir, timeout=60, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "checkout", parsed_info["sha"]],
                cwd=agent_dir, timeout=30, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        elif resource in ("blob", "tree") and "branch" in parsed_info:
            subprocess.run(
                ["git", "checkout", parsed_info["branch"]],
                cwd=agent_dir, timeout=30, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        elif resource in ("blob", "resolve", "tree") and "revision" in parsed_info:
            subprocess.run(
                ["git", "checkout", parsed_info["revision"]],
                cwd=agent_dir, timeout=30, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
    except Exception:
        pass  # Best effort


def capture_diff(agent_dir):
    """Capture the cumulative git diff for an agent's working directory.

    Stages all changes then diffs against HEAD, excluding opencode
    infrastructure files so only the agent's actual work appears.
    """
    subprocess.run(
        ["git", "add", "-A"],
        cwd=agent_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    result = subprocess.run(
        [
            "git", "diff", "HEAD", "--",
            ".",
            ":(exclude)opencode.json",
            ":(exclude).opencode",
            ":(exclude).xdg_data",
            ":(exclude).tmp",
        ],
        cwd=agent_dir, capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")[:100_000]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_all_rounds(rounds):
    """Format all agent rounds for display.

    Each round shows the user bubble, the model bubble, and — when the
    agent produced file changes — the cumulative git diff up to that
    round.  Rendering the diff per-round (rather than once at the end)
    means successive responses always re-check and refresh the diff.
    Model output is converted from Markdown to HTML via the markdown library.
    """
    SEPARATOR = (
        "<div style='text-align: center; color: #888888; margin: 16px 0; "
        "border-top: 1px solid #dddddd; padding-top: 10px;'>"
        "<em>&#8213; Follow-up &#8213;</em></div>\n"
    )
    formatted = ""
    for i, r in enumerate(rounds):
        output_html = md_lib.markdown(r["output"], extensions=["fenced_code", "tables", "nl2br"])
        if i > 0:
            # Show follow-up separator and the user's follow-up prompt
            prompt_html = md_lib.markdown(r["prompt"], extensions=["fenced_code", "tables", "nl2br"])
            formatted += SEPARATOR
            formatted += (
                f"<div style='color: #0066cc; background-color: #f0f7ff; "
                f"padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                f"<strong>User:</strong> {prompt_html}</div>\n"
            )
        # The initial user query is already shown above the panel, so skip it here.
        formatted += (
            f"<div style='color: #006633; background-color: #f0fff0; "
            f"padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
            f"<strong>Model:</strong> {output_html}</div>\n"
        )
        if r.get("diff"):
            diff_html = md_lib.markdown(
                f"**Git Diff (cumulative after round {i + 1}):**\n"
                f"```diff\n{r['diff']}\n```",
                extensions=["fenced_code"],
            )
            formatted += diff_html + "\n"
    return formatted


# ---------------------------------------------------------------------------
# HF data I/O
# ---------------------------------------------------------------------------

def save_content_to_hf(data, repo_name, file_name, token=None):
    """Save content to Hugging Face repository."""
    json_content = json.dumps(data, indent=4).encode("utf-8")
    file_like_object = io.BytesIO(json_content)
    filename = f"{file_name}.json"

    if token is None:
        token = os.getenv("HF_TOKEN") or HfApi().token
    if token is None:
        raise ValueError("Please log in to Hugging Face to submit votes.")

    upload_file(
        path_or_fileobj=file_like_object,
        path_in_repo=filename,
        repo_id=repo_name,
        repo_type="dataset",
        token=token,
    )


def is_file_within_time_frame(file_path, days):
    try:
        timestamp_str = file_path.split("/")[-1].split(".")[0]
        file_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        time_diff = datetime.now() - file_datetime
        return time_diff.days <= days
    except:
        return False


def load_content_from_hf(repo_name, file_name):
    """Read content from a Hugging Face repository within the configured time frame."""
    data = []
    try:
        api = HfApi()
        file_prefix = f"{file_name}/"
        for file in api.list_repo_files(repo_id=repo_name, repo_type="dataset"):
            if not file.startswith(file_prefix):
                continue
            if not is_file_within_time_frame(file, LEADERBOARD_UPDATE_TIME_FRAME_DAYS):
                continue
            local_path = hf_hub_download(
                repo_id=repo_name, filename=file, repo_type="dataset"
            )
            with open(local_path, "r") as f:
                entry = json.load(f)
                entry["timestamp"] = file.split("/")[-1].split(".")[0]
                data.append(entry)
        return data
    except:
        raise Exception("Error loading feedback data from Hugging Face repository.")


# ---------------------------------------------------------------------------
# Leaderboard computation
# ---------------------------------------------------------------------------

def get_leaderboard_data(vote_entry=None, use_cache=True):
    if use_cache:
        try:
            cached_path = hf_hub_download(
                repo_id=LEADERBOARD_REPO,
                filename=f'{LEADERBOARD_FILE}.json',
                repo_type="dataset",
            )
            with open(cached_path, "r") as f:
                leaderboard_data = pd.read_json(f)
                round_cols = {
                    "Elo Score": 2,
                    "Win Rate": 2,
                    "Conversation Efficiency Index": 2,
                    "Consistency Score": 2,
                    "Bradley-Terry Coefficient": 2,
                    "Eigenvector Centrality Value": 2,
                    "Newman Modularity Score": 2,
                    "PageRank Score": 2,
                }
                for col, decimals in round_cols.items():
                    if col in leaderboard_data.columns:
                        leaderboard_data[col] = pd.to_numeric(leaderboard_data[col], errors="coerce").round(decimals)
                return leaderboard_data
        except Exception as e:
            print(f"No cached leaderboard found, computing from votes...")

    data = load_content_from_hf(VOTE_REPO, LEADERBOARD_FILE)
    vote_df = pd.DataFrame(data)

    if vote_entry is not None:
        vote_df = pd.concat([vote_df, pd.DataFrame([vote_entry])], ignore_index=True)

    if vote_df.empty:
        return pd.DataFrame(
            columns=[
                "Rank", "Model", "Organization", "Elo Score", "Win Rate",
                "Conversation Efficiency Index", "Consistency Score",
                "Bradley-Terry Coefficient", "Eigenvector Centrality Value",
                "Newman Modularity Score", "PageRank Score",
            ]
        )

    # Load conversation data and merge for CEI/MCS computation
    conversation_data = load_content_from_hf(CONVERSATION_REPO, LEADERBOARD_FILE)
    conversation_df = pd.DataFrame(conversation_data)

    all_df = pd.merge(
        vote_df, conversation_df, on=["timestamp", "left", "right"], how="inner"
    )

    # Compute CEI and MCS from merged data
    model_stats = {}

    for _, row in all_df.iterrows():
        left_model = row["left"]
        right_model = row["right"]
        is_self_match = left_model == right_model

        for model in [left_model, right_model]:
            if model not in model_stats:
                model_stats[model] = {
                    "cei_sum": 0,
                    "cei_max": 0,
                    "self_matches": 0,
                    "self_draws": 0,
                }

        if is_self_match:
            model_stats[left_model]["self_matches"] += 1
            if row["winner"] == "both_bad" or row["winner"] == "tie":
                model_stats[left_model]["self_draws"] += 1
            continue

        match row["winner"]:
            case "left":
                left_score = 1
                right_score = -1
            case "right":
                left_score = -1
                right_score = 1
            case "tie":
                left_score = 0.3
                right_score = 0.3
            case "both_bad":
                left_score = -0.3
                right_score = -0.3

        # Round count = number of opencode invocations per side
        left_rounds = len(row.get("left_rounds", [])) or 1
        right_rounds = len(row.get("right_rounds", [])) or 1

        model_stats[left_model]["cei_max"] += 1 / left_rounds
        model_stats[right_model]["cei_max"] += 1 / right_rounds
        model_stats[left_model]["cei_sum"] += left_score / left_rounds
        model_stats[right_model]["cei_sum"] += right_score / right_rounds

    # Map vote winners for evalica
    vote_df["winner"] = vote_df["winner"].map(
        {
            "left": evalica.Winner.X,
            "right": evalica.Winner.Y,
            "tie": evalica.Winner.Draw,
            "both_bad": evalica.Winner.Draw,
        }
    )

    # Calculate scores using various metrics
    avr_result = evalica.average_win_rate(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0,
    )
    bt_result = evalica.bradley_terry(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )
    newman_result = evalica.newman(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )
    eigen_result = evalica.eigen(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )
    elo_result = evalica.elo(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )
    pagerank_result = evalica.pagerank(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )

    # Clean up inf/NaN values
    avr_scores = avr_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    bt_scores = bt_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    newman_scores = newman_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    eigen_scores = eigen_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    elo_scores = elo_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    pagerank_scores = pagerank_result.scores.replace([float("inf"), float("-inf")], float("nan"))

    # Calculate CEI results
    cei_result = {}
    for model in elo_scores.index:
        if model in model_stats and model_stats[model]["cei_max"] > 0:
            cei_result[model] = round(
                model_stats[model]["cei_sum"] / model_stats[model]["cei_max"], 2
            )
        else:
            cei_result[model] = None
    cei_result = pd.Series(cei_result)

    # Calculate MCS results
    mcs_result = {}
    for model in elo_scores.index:
        if model in model_stats and model_stats[model]["self_matches"] > 0:
            mcs_result[model] = round(
                model_stats[model]["self_draws"] / model_stats[model]["self_matches"], 2
            )
        else:
            mcs_result[model] = None
    mcs_result = pd.Series(mcs_result)
    organization_values = [model_organization.get(model, "") for model in elo_scores.index]

    leaderboard_data = pd.DataFrame(
        {
            "Model": [name.split(": ", 1)[-1] for name in elo_scores.index],
            "Organization": organization_values,
            "Elo Score": elo_scores.values,
            "Win Rate": avr_scores.values,
            "Conversation Efficiency Index": cei_result.values,
            "Consistency Score": mcs_result.values,
            "Bradley-Terry Coefficient": bt_scores.values,
            "Eigenvector Centrality Value": eigen_scores.values,
            "Newman Modularity Score": newman_scores.values,
            "PageRank Score": pagerank_scores.values,
        }
    )

    round_cols = {
        "Elo Score": 2,
        "Win Rate": 2,
        "Bradley-Terry Coefficient": 2,
        "Eigenvector Centrality Value": 2,
        "Newman Modularity Score": 2,
        "PageRank Score": 2,
    }
    for col, decimals in round_cols.items():
        if col in leaderboard_data.columns:
            leaderboard_data[col] = pd.to_numeric(leaderboard_data[col], errors="coerce").round(decimals)

    leaderboard_data["Rank"] = (
        leaderboard_data["Elo Score"].rank(method="min", ascending=False).astype(int)
    )
    leaderboard_data = leaderboard_data[
        ["Rank"] + [col for col in leaderboard_data.columns if col != "Rank"]
    ]

    if vote_entry is not None:
        try:
            json_content = leaderboard_data.to_json(orient="records", indent=4).encode("utf-8")
            file_like_object = io.BytesIO(json_content)
            upload_file(
                path_or_fileobj=file_like_object,
                path_in_repo=f'{LEADERBOARD_FILE}.json',
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset",
                token=HfApi().token,
            )
        except Exception as e:
            print(f"Failed to save leaderboard cache: {e}")

    return leaderboard_data


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

def guardrail_check_se_relevance(user_input):
    """Check if the user input is SE-related using a lightweight LLM classifier."""
    system_message = {
        "role": "system",
        "content": (
            "You are a classifier that decides if a user's question is relevant to software engineering. "
            "If the question is about software engineering concepts, tools, processes, or code, respond with 'Yes'. "
            "Otherwise, respond with 'No'."
        ),
    }
    user_message = {"role": "user", "content": user_input}
    try:
        response = openai_client.chat.completions.create(
            model="openai/gpt-oss-safeguard-20b", messages=[system_message, user_message]
        )
        classification = response.choices[0].message.content.strip().lower()
        return classification.startswith("yes")
    except Exception as e:
        print(f"Guardrail check failed: {e}")
        return True  # fail open


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def toggle_submit_button(text):
    if not text or text.strip() == "":
        return gr.update(interactive=False)
    else:
        return gr.update(interactive=True)


def check_auth_on_load(request: gr.Request):
    """Check if user is already authenticated when page loads."""
    token = os.getenv("HF_TOKEN") or HfApi().token
    # Also try to get the token from the Gradio OAuth session (set after gr.LoginButton login)
    if token is None and hasattr(request, 'oauth_token') and request.oauth_token is not None:
        try:
            token = request.oauth_token.token
        except AttributeError:
            token = str(request.oauth_token) if request.oauth_token else None
    is_authenticated = (hasattr(request, 'username') and request.username is not None and request.username != "")

    if is_authenticated or token:
        return (
            gr.update(interactive=True),   # repo_url
            gr.update(interactive=True),   # shared_input
            gr.update(interactive=False),  # send_first (disabled until text entered)
            gr.update(interactive=True),   # feedback
            gr.update(interactive=True),   # submit_feedback_btn
            gr.update(visible=False),      # hint_markdown
            gr.update(visible=True),       # login_button
            token,                         # oauth_token
        )
    else:
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=True),
            gr.update(visible=True),
            None,
        )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

warnings.filterwarnings('ignore', category=DeprecationWarning, message=".*'theme' parameter.*")
with gr.Blocks(title="SWE-Model-Arena", theme=gr.themes.Soft()) as app:
    user_authenticated = gr.State(False)
    models_state = gr.State({})
    conversation_state = gr.State({})
    oauth_token = gr.State(None)

    with gr.Tab("🏆Leaderboard"):
        gr.Markdown("# 🏆 LLM4ASE Leaderboard")
        gr.Markdown(
            "Community-driven evaluation of LLMs on real agentic coding tasks, "
            "powered by [opencode](https://opencode.ai)"
        )
        gr.Markdown(
            "*SWE-Model-Arena pits LLMs head-to-head in blind agentic coding comparisons. "
            "Each model drives [opencode](https://github.com/opencode-ai/opencode) to read files, write code, "
            "runs commands, and produces real git diffs — identical scaffold, different brain. "
            "Community votes determine the rankings. "
            "For technical details, check out our [paper](https://arxiv.org/abs/2502.01860).*"
        )

        leaderboard_component = Leaderboard(
            value=get_leaderboard_data(use_cache=True),
            select_columns=[
                "Rank", "Model", "Organization", "Elo Score",
                "Conversation Efficiency Index", "Consistency Score",
            ],
            search_columns=["Model"],
            filter_columns=[
                ColumnFilter("Elo Score", min=800, max=1600, default=[800, 1600], type="slider", label="Elo Score"),
                ColumnFilter("Win Rate", min=0, max=1, default=[0, 1], type="slider", label="Win Rate"),
                ColumnFilter("Conversation Efficiency Index", min=0, max=1, default=[0, 1], type="slider", label="Conversation Efficiency Index"),
                ColumnFilter("Consistency Score", min=0, max=1, default=[0, 1], type="slider", label="Consistency Score"),
            ],
            datatype=[
                "number", "str", "str", "number", "number", "number",
                "number", "number", "number", "number", "number",
            ],
        )

        gr.Markdown("---")
        gr.Markdown(
            """
            Made with ❤️ for SWE-Model-Arena. If this work is useful to you, please consider citing our vision paper:
            ```
            @inproceedings{zhao2025se,
            title={SE Arena: An Interactive Platform for Evaluating Foundation Models in Software Engineering},
            author={Zhao, Zhimin},
            booktitle={2025 IEEE/ACM Second International Conference on AI Foundation Models and Software Engineering (Forge)},
            pages={78--81},
            year={2025},
            organization={IEEE}
            }
            ```
            """
        )

    with gr.Tab("⚔️Arena"):
        gr.Markdown("# ⚔️ SWE-Model-Arena")
        gr.Markdown("Blind head-to-head tool-calling model comparison — same scaffold (opencode), different tool-calling LLM")

        gr.Markdown("### 📜 How It Works")
        gr.Markdown(
            f"""
            - **Blind Comparison**: Submit a coding task — two randomly selected tool-calling LLMs will tackle it independently (up to {len(active_models)} models).
            - **Same Scaffold, Different Brain**: Both models run on [opencode](https://github.com/opencode-ai/opencode) — an agentic coding engine that reads files, writes code, and runs commands. Only the underlying LLM differs.
            - **Real Diffs**: Each model works in its own isolated git repo. You see the actual code changes, not just chat responses.
            - **Multi-round & Vote**: Send follow-up instructions to either side, then vote for the better model. Fair play — votes count only while identities stay hidden.
            """
        )
        gr.Markdown(f"*Note: Due to resource constraints, sessions that take longer than {AGENT_TIMEOUT} seconds will be terminated.*")

        gr.Markdown("---")
        
        with gr.Row():
            # Define the markdown text with or without the hint string
            markdown_text = "### Please sign in first to vote!"
            if SHOW_HINT_STRING:
                markdown_text += f"\n*{HINT_STRING}*"
            hint_markdown = gr.Markdown(markdown_text)
            with gr.Column():
                login_button = gr.LoginButton(
                    "Sign in with Hugging Face", elem_id="oauth-button"
                )

        guardrail_message = gr.Markdown("", visible=False, elem_id="guardrail-message")

        repo_url = gr.Textbox(
            show_label=False,
            placeholder="Optional: Enter any GitHub, GitLab, or Hugging Face URL.",
            lines=1,
            interactive=False,
        )

        shared_input = gr.Textbox(
            show_label=False,
            placeholder="Enter your task for both models here.",
            lines=2,
            interactive=False,
        )
        send_first = gr.Button("Submit", visible=True, interactive=False)

        shared_input.change(
            fn=toggle_submit_button, inputs=shared_input, outputs=send_first
        )

        user_prompt_md = gr.Markdown(value="", visible=False)

        with gr.Row():
            response_a_title = gr.Markdown(value="", visible=False)
            response_b_title = gr.Markdown(value="", visible=False)

        with gr.Row():
            response_a = gr.Markdown(label="Response from Model A")
            response_b = gr.Markdown(label="Response from Model B")

        # Timeout popup
        with gr.Row(visible=False) as timeout_popup:
            timeout_message = gr.Markdown(
                f"### Timeout\n\nOne of the models did not respond within {AGENT_TIMEOUT} seconds. Please try again."
            )
            close_popup_btn = gr.Button("Okay")

        def close_timeout_popup():
            shared_input_state = gr.update(interactive=True)
            send_first_state = toggle_submit_button(shared_input.value)
            model_a_input_state = gr.update(interactive=True)
            model_a_send_state = toggle_submit_button(model_a_input.value)
            model_b_input_state = gr.update(interactive=True)
            model_b_send_state = toggle_submit_button(model_b_input.value)
            repo_url_state = gr.update(interactive=True)
            return (
                gr.update(visible=False),
                shared_input_state,
                send_first_state,
                model_a_input_state,
                model_a_send_state,
                model_b_input_state,
                model_b_send_state,
                repo_url_state,
            )

        # Multi-round inputs, initially hidden
        with gr.Row(visible=False) as multi_round_inputs:
            model_a_input = gr.Textbox(label="Model A Input", lines=1)
            model_a_send = gr.Button("Send to Model A", interactive=False)
            model_b_input = gr.Textbox(label="Model B Input", lines=1)
            model_b_send = gr.Button("Send to Model B", interactive=False)

        model_a_input.change(
            fn=toggle_submit_button, inputs=model_a_input, outputs=model_a_send
        )
        model_b_input.change(
            fn=toggle_submit_button, inputs=model_b_input, outputs=model_b_send
        )

        close_popup_btn.click(
            close_timeout_popup,
            inputs=[],
            outputs=[
                timeout_popup, shared_input, send_first,
                model_a_input, model_a_send,
                model_b_input, model_b_send,
                repo_url,
            ],
        )

        # -- Handlers --

        def disable_first_submit_ui():
            return (
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False, value="Processing..."),
            )

        def _cleanup_agent_resources(conversation_state):
            """Stop opencode servers and clean up temp directories."""
            for port_key in ["left_port", "right_port"]:
                if port_key in conversation_state:
                    stop_opencode_server(conversation_state[port_key])
            for dir_key in ["left_dir", "right_dir"]:
                if dir_key in conversation_state:
                    shutil.rmtree(conversation_state[dir_key], ignore_errors=True)

        def update_model_titles_and_responses(
            repo_url, user_input, models_state, conversation_state
        ):
            # Guardrail check (skip if URL provided)
            if not repo_url and not guardrail_check_se_relevance(user_input):
                yield (
                    gr.update(value="### Oops! Try asking something about software engineering. Thanks!", visible=True),
                    gr.update(value="", interactive=True, visible=True),
                    gr.update(value="", interactive=True, visible=True),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True, interactive=True, value="Submit"),
                    gr.update(interactive=True),
                    models_state,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(visible=False),
                )
                return

            # Fetch repo context
            repo_info = fetch_url_content(repo_url)

            # Create temp dirs with UUID names to avoid stale-dir collisions
            session_id = uuid.uuid4().hex
            base_tmp = tempfile.gettempdir()
            left_dir = os.path.join(base_tmp, f"agent_left_{session_id}")
            right_dir = os.path.join(base_tmp, f"agent_right_{session_id}")
            os.makedirs(left_dir, exist_ok=True)
            os.makedirs(right_dir, exist_ok=True)

            # Build per-agent prompts — each includes its concrete workspace
            # path so the agent knows exactly where it is allowed to operate.
            left_prompt = build_prompt(user_input, repo_info, agent_dir=left_dir)
            right_prompt = build_prompt(user_input, repo_info, agent_dir=right_dir)

            # Allocate ports for opencode servers
            left_port = find_free_port()
            right_port = find_free_port()

            display_content = f"### Your Query:\n\n{user_input}"
            if repo_info:
                display_content += f"\n\n### Repo-related URL:\n\n{repo_url}"

            def _cleanup_on_error():
                """Stop servers and remove temp dirs on failure."""
                stop_opencode_server(left_port)
                stop_opencode_server(right_port)
                shutil.rmtree(left_dir, ignore_errors=True)
                shutil.rmtree(right_dir, ignore_errors=True)

            try:
                # Git init or clone in each temp dir
                for d in [left_dir, right_dir]:
                    if repo_url and repo_url.strip():
                        clone_repo(repo_url, d)
                    else:
                        subprocess.run(["git", "init"], cwd=d, capture_output=True)
                        # Create an initial empty commit so HEAD always exists.
                        # Without this, `git diff --cached` returns nothing on an
                        # unborn branch, making all diffs appear empty.
                        subprocess.run(
                            ["git", "-c", "user.name=arena", "-c", "user.email=arena@localhost",
                             "commit", "--allow-empty", "-m", "init"],
                            cwd=d, capture_output=True,
                        )

                # Pre-select two distinct preferred models (mirrors run_first_round_with_retry).
                global_deadline = time.time() + BATTLE_TIMEOUT
                left_preferred = random.choice(active_models)
                right_candidates = [m for m in active_models if m != left_preferred]
                right_preferred = random.choice(right_candidates) if right_candidates else left_preferred

                # Run both agents concurrently in threads so we can yield
                # partial results as soon as the first agent finishes.
                partial = {
                    "left_name": None, "left_result": None, "left_diff": None,
                    "right_name": None, "right_result": None, "right_diff": None,
                }
                futures_map = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futures_map[executor.submit(
                        _run_agent_in_thread,
                        left_dir, left_port, left_prompt, left_preferred, global_deadline,
                    )] = "left"
                    futures_map[executor.submit(
                        _run_agent_in_thread,
                        right_dir, right_port, right_prompt, right_preferred, global_deadline,
                    )] = "right"

                    for future in concurrent.futures.as_completed(futures_map):
                        side = futures_map[future]
                        name, result = future.result()

                        if not result.get("ok"):
                            err = result.get("error", "unknown")
                            label = "A" if side == "left" else "B"
                            print(f"[Arena] Agent {label} failed: {err}")
                            result["output"] = f"**Model error:** {err}"

                        diff = capture_diff(left_dir if side == "left" else right_dir)
                        partial[f"{side}_name"] = name
                        partial[f"{side}_result"] = result
                        partial[f"{side}_diff"] = diff

                        both_done = (
                            partial["left_result"] is not None
                            and partial["right_result"] is not None
                        )

                        left_rounds = (
                            [{"prompt": left_prompt,
                              "output": partial["left_result"]["output"],
                              "diff": partial["left_diff"]}]
                            if partial["left_result"] else None
                        )
                        right_rounds = (
                            [{"prompt": right_prompt,
                              "output": partial["right_result"]["output"],
                              "diff": partial["right_diff"]}]
                            if partial["right_result"] else None
                        )

                        display_a = (
                            format_all_rounds(left_rounds)
                            if left_rounds else "\u23f3 *Waiting for model...*"
                        )
                        display_b = (
                            format_all_rounds(right_rounds)
                            if right_rounds else "\u23f3 *Waiting for model...*"
                        )

                        if both_done:
                            # Final state — update shared state and enable multi-round.
                            models_state.clear()
                            models_state.update({
                                "left": partial["left_name"],
                                "right": partial["right_name"],
                            })
                            conversation_state.clear()
                            conversation_state.update({
                                "left": partial["left_name"],
                                "right": partial["right_name"],
                                "url": repo_url or "",
                                "left_dir": left_dir, "right_dir": right_dir,
                                "left_port": left_port, "right_port": right_port,
                                "left_session_id": partial["left_result"].get("session_id"),
                                "right_session_id": partial["right_result"].get("session_id"),
                                "left_rounds": left_rounds,
                                "right_rounds": right_rounds,
                            })
                            yield (
                                gr.update(visible=False),                          # [0] guardrail_message
                                gr.update(interactive=True, visible=False),        # [1] shared_input
                                gr.update(interactive=True, visible=False),        # [2] repo_url
                                gr.update(value=display_content, visible=True),    # [3] user_prompt_md
                                gr.update(value="### Model A", visible=True),      # [4] response_a_title
                                gr.update(value="### Model B", visible=True),      # [5] response_b_title
                                gr.update(value=display_a),                        # [6] response_a
                                gr.update(value=display_b),                        # [7] response_b
                                gr.update(visible=True),                           # [8] multi_round_inputs
                                gr.update(visible=True),                           # [9] vote_panel
                                gr.update(visible=False, value="Submit"),          # [10] send_first
                                gr.update(interactive=True),                       # [11] feedback
                                models_state,                                      # [12] models_state
                                conversation_state,                                # [13] conversation_state
                                gr.update(visible=False),                          # [14] timeout_popup
                                toggle_submit_button(""),                          # [15] model_a_send
                                toggle_submit_button(""),                          # [16] model_b_send
                                gr.update(visible=False),                          # [17] thanks_message
                            )
                        else:
                            # Partial state — first agent done, second still running.
                            # Show vote panel early so the user can vote based on
                            # latency / first impression. Multi-round stays hidden
                            # until both agents are done and state is fully populated.
                            yield (
                                gr.update(visible=False),                          # [0] guardrail_message
                                gr.update(interactive=True, visible=False),        # [1] shared_input
                                gr.update(interactive=True, visible=False),        # [2] repo_url
                                gr.update(value=display_content, visible=True),    # [3] user_prompt_md
                                gr.update(value="### Model A", visible=True),      # [4] response_a_title
                                gr.update(value="### Model B", visible=True),      # [5] response_b_title
                                gr.update(value=display_a),                        # [6] response_a
                                gr.update(value=display_b),                        # [7] response_b
                                gr.update(visible=False),                          # [8] multi_round_inputs (wait for both)
                                gr.update(visible=True),                           # [9] vote_panel (early!)
                                gr.update(visible=False, value="Submit"),          # [10] send_first
                                gr.update(interactive=True),                       # [11] feedback
                                models_state,                                      # [12] models_state (not yet complete)
                                conversation_state,                                # [13] conversation_state (not yet complete)
                                gr.update(visible=False),                          # [14] timeout_popup
                                toggle_submit_button(""),                          # [15] model_a_send
                                toggle_submit_button(""),                          # [16] model_b_send
                                gr.update(visible=False),                          # [17] thanks_message
                            )

            except TimeoutError as e:
                _cleanup_on_error()
                yield (
                    gr.update(visible=False),                               # [0] guardrail_message
                    gr.update(interactive=True, visible=True),              # [1] shared_input
                    gr.update(interactive=True, visible=True),              # [2] repo_url
                    gr.update(value="", visible=False),                     # [3] user_prompt_md
                    gr.update(value="", visible=False),                     # [4] response_a_title
                    gr.update(value="", visible=False),                     # [5] response_b_title
                    gr.update(value=""),                                    # [6] response_a
                    gr.update(value=""),                                    # [7] response_b
                    gr.update(visible=False),                               # [8] multi_round_inputs
                    gr.update(visible=False),                               # [9] vote_panel
                    gr.update(visible=True, interactive=True, value="Submit"),  # [10] send_first
                    gr.update(interactive=False),                           # [11] feedback
                    models_state,                                           # [12] models_state
                    conversation_state,                                     # [13] conversation_state
                    gr.update(visible=True),                                # [14] timeout_popup
                    gr.update(interactive=False),                           # [15] model_a_send
                    gr.update(interactive=False),                           # [16] model_b_send
                    gr.update(visible=False),                               # [17] thanks_message
                )
                return
            except Exception as e:
                _cleanup_on_error()
                yield (
                    gr.update(value=f"### Error: {str(e)}", visible=True),  # [0] guardrail_message
                    gr.update(interactive=True, visible=True),              # [1] shared_input
                    gr.update(interactive=True, visible=True),              # [2] repo_url
                    gr.update(value="", visible=False),                     # [3] user_prompt_md
                    gr.update(value="", visible=False),                     # [4] response_a_title
                    gr.update(value="", visible=False),                     # [5] response_b_title
                    gr.update(value=""),                                    # [6] response_a
                    gr.update(value=""),                                    # [7] response_b
                    gr.update(visible=False),                               # [8] multi_round_inputs
                    gr.update(visible=False),                               # [9] vote_panel
                    gr.update(visible=True, interactive=True, value="Submit"),  # [10] send_first
                    gr.update(interactive=False),                           # [11] feedback
                    models_state,                                           # [12] models_state
                    conversation_state,                                     # [13] conversation_state
                    gr.update(visible=False),                               # [14] timeout_popup
                    gr.update(interactive=False),                           # [15] model_a_send
                    gr.update(interactive=False),                           # [16] model_b_send
                    gr.update(visible=False),                               # [17] thanks_message
                )
                return

        # Feedback panel, initially hidden
        with gr.Column(visible=False) as vote_panel:
            gr.Markdown("### Which model do you prefer?")
            with gr.Row():
                feedback = gr.Radio(
                    choices=["Model A", "Model B", "Tie", "Tie (Both Bad)"],
                    show_label=False,
                    value="Tie",
                    interactive=False,
                )
                submit_feedback_btn = gr.Button("Submit Feedback", interactive=False)

        thanks_message = gr.Markdown(value="", visible=False)

        def hide_thanks_message():
            return gr.update(visible=False)

        def handle_login(request: gr.Request):
            token = os.getenv("HF_TOKEN") or HfApi().token
            is_authenticated = hasattr(request, 'username') and request.username

            if is_authenticated or token:
                return (
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    token,
                )
            else:
                return (
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(visible=True, value="## Please sign in with Hugging Face!\nClick the 'Sign in with Hugging Face' button above."),
                    gr.update(visible=True),
                    None,
                )

        # First round handling
        send_first.click(
            fn=hide_thanks_message, inputs=[], outputs=[thanks_message]
        ).then(
            fn=disable_first_submit_ui,
            inputs=[],
            outputs=[guardrail_message, shared_input, repo_url, send_first],
        ).then(
            fn=update_model_titles_and_responses,
            inputs=[repo_url, shared_input, models_state, conversation_state],
            outputs=[
                guardrail_message, shared_input, repo_url,
                user_prompt_md, response_a_title, response_b_title,
                response_a, response_b,
                multi_round_inputs, vote_panel,
                send_first, feedback,
                models_state, conversation_state,
                timeout_popup, model_a_send, model_b_send,
                thanks_message,
            ],
        )

        # -- Follow-up round handlers --

        def disable_model_a_ui():
            return (
                gr.update(interactive=False),
                gr.update(interactive=False, value="Processing..."),
            )

        def handle_model_a_send(user_input, models_state, conversation_state):
            """Handle a follow-up message for Model A.

            Always regenerates the full display from conversation_state so the
            complete history is rendered correctly from raw data, avoiding any
            issues with reading processed HTML back from the Markdown component.
            """
            try:
                port = conversation_state["left_port"]
                session_id = conversation_state["left_session_id"]
                model_id = model_name_to_id[conversation_state["left"]]

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    run_agent(port, model_id, user_input, session_id=session_id)
                )
                loop.close()

                # Show error/timeout output in the panel instead of crashing
                output = result.get("output", "")
                if not result.get("ok"):
                    err = result.get("error", "Model failed")
                    output = output or f"**Model error:** {err}"

                conversation_state["left_session_id"] = result.get("session_id", session_id)
                diff = capture_diff(conversation_state["left_dir"])
                conversation_state["left_rounds"].append({
                    "prompt": user_input, "output": output, "diff": diff,
                })

                # Regenerate full display from state so history is always correct.
                formatted = format_all_rounds(conversation_state["left_rounds"])
                return (
                    formatted,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(value="", interactive=True),
                    gr.update(interactive=False, value="Send to Model A"),
                )
            except TimeoutError:
                formatted = format_all_rounds(conversation_state.get("left_rounds", []))
                return (
                    formatted,
                    conversation_state,
                    gr.update(visible=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True, value="Send to Model A"),
                )
            except Exception as e:
                err_round = {"prompt": user_input, "output": f"**Error:** {str(e)}", "diff": ""}
                display_rounds = conversation_state.get("left_rounds", []) + [err_round]
                formatted = format_all_rounds(display_rounds)
                return (
                    formatted,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    gr.update(interactive=True, value="Send to Model A"),
                )

        def disable_model_b_ui():
            return (
                gr.update(interactive=False),
                gr.update(interactive=False, value="Processing..."),
            )

        def handle_model_b_send(user_input, models_state, conversation_state):
            """Handle a follow-up message for Model B.

            Always regenerates the full display from conversation_state so the
            complete history is rendered correctly from raw data, avoiding any
            issues with reading processed HTML back from the Markdown component.
            """
            try:
                port = conversation_state["right_port"]
                session_id = conversation_state["right_session_id"]
                model_id = model_name_to_id[conversation_state["right"]]

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    run_agent(port, model_id, user_input, session_id=session_id)
                )
                loop.close()

                # Show error/timeout output in the panel instead of crashing
                output = result.get("output", "")
                if not result.get("ok"):
                    err = result.get("error", "Model failed")
                    output = output or f"**Model error:** {err}"

                conversation_state["right_session_id"] = result.get("session_id", session_id)
                diff = capture_diff(conversation_state["right_dir"])
                conversation_state["right_rounds"].append({
                    "prompt": user_input, "output": output, "diff": diff,
                })

                # Regenerate full display from state so history is always correct.
                formatted = format_all_rounds(conversation_state["right_rounds"])
                return (
                    formatted,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(value="", interactive=True),
                    gr.update(interactive=False, value="Send to Model B"),
                )
            except TimeoutError:
                formatted = format_all_rounds(conversation_state.get("right_rounds", []))
                return (
                    formatted,
                    conversation_state,
                    gr.update(visible=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True, value="Send to Model B"),
                )
            except Exception as e:
                err_round = {"prompt": user_input, "output": f"**Error:** {str(e)}", "diff": ""}
                display_rounds = conversation_state.get("right_rounds", []) + [err_round]
                formatted = format_all_rounds(display_rounds)
                return (
                    formatted,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    gr.update(interactive=True, value="Send to Model B"),
                )

        model_a_send.click(
            fn=disable_model_a_ui,
            inputs=[],
            outputs=[model_a_input, model_a_send],
        ).then(
            fn=handle_model_a_send,
            inputs=[model_a_input, models_state, conversation_state],
            outputs=[response_a, conversation_state, timeout_popup, model_a_input, model_a_send],
        )
        model_b_send.click(
            fn=disable_model_b_ui,
            inputs=[],
            outputs=[model_b_input, model_b_send],
        ).then(
            fn=handle_model_b_send,
            inputs=[model_b_input, models_state, conversation_state],
            outputs=[response_b, conversation_state, timeout_popup, model_b_input, model_b_send],
        )

        # -- Vote handler --

        def submit_feedback(vote, models_state, conversation_state, token):
            """Save the vote and conversation, then reveal model identities.

            The vote is persisted BEFORE identities are returned to the
            UI, ensuring the user cannot be influenced by knowing which
            model is which.
            """
            match vote:
                case "Model A":
                    winner = "left"
                case "Model B":
                    winner = "right"
                case "Tie":
                    winner = "tie"
                case _:
                    winner = "both_bad"

            file_name = f"{LEADERBOARD_FILE}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            vote_entry = {
                "left": models_state["left"],
                "right": models_state["right"],
                "winner": winner,
            }
            save_content_to_hf(vote_entry, VOTE_REPO, file_name, token)

            # Strip context from first round prompts before saving
            for side in ["left_rounds", "right_rounds"]:
                if conversation_state.get(side):
                    conversation_state[side][0]["prompt"] = strip_context(
                        conversation_state[side][0]["prompt"]
                    )

            # Save conversation (only persistable fields)
            conv_data = {
                "left": conversation_state["left"],
                "right": conversation_state["right"],
                "url": conversation_state.get("url", ""),
                "left_rounds": conversation_state["left_rounds"],
                "right_rounds": conversation_state["right_rounds"],
            }
            save_content_to_hf(conv_data, CONVERSATION_REPO, file_name, token)

            # Build the reveal message AFTER vote is saved
            model_a_name = models_state.get("left", "Unknown")
            model_b_name = models_state.get("right", "Unknown")
            thanks_text = (
                "## Thanks for your vote! Identities revealed below.\n"
                f"**Model A:** {model_a_name}\n\n"
                f"**Model B:** {model_b_name}"
            )

            # Clean up temp dirs
            _cleanup_agent_resources(conversation_state)

            models_state.clear()
            conversation_state.clear()

            return (
                gr.update(value="", interactive=True, visible=True),    # [0] shared_input
                gr.update(value="", interactive=True, visible=True),    # [1] repo_url
                gr.update(value="", visible=False),                     # [2] user_prompt_md
                gr.update(value="", visible=False),                     # [3] response_a_title
                gr.update(value="", visible=False),                     # [4] response_b_title
                gr.update(value=""),                                    # [5] response_a
                gr.update(value=""),                                    # [6] response_b
                gr.update(visible=False),                               # [7] multi_round_inputs
                gr.update(visible=False),                               # [8] vote_panel
                gr.update(value="Submit", interactive=True, visible=True),  # [9] send_first
                gr.update(value="Tie", interactive=True),               # [10] feedback
                get_leaderboard_data(vote_entry, use_cache=False),      # [11] leaderboard
                gr.update(value=thanks_text, visible=True),             # [12] thanks_message
            )

        submit_feedback_btn.click(
            submit_feedback,
            inputs=[feedback, models_state, conversation_state, oauth_token],
            outputs=[
                shared_input, repo_url, user_prompt_md,
                response_a_title, response_b_title,
                response_a, response_b,
                multi_round_inputs, vote_panel,
                send_first, feedback,
                leaderboard_component,
                thanks_message,
            ],
        )

        gr.Markdown("---")
        gr.Markdown("### Terms of Service")
        gr.Markdown(
            """
            *Users are required to agree to the following terms before using the service:*

            - The service is a **research preview**. It only provides limited safety measures and may generate offensive content.
            - It must not be used for any **illegal, harmful, violent, racist, or sexual** purposes.
            - Please do not upload any **private** information.
            - The service collects user dialogue data, including both text and images, and reserves the right to distribute it under a **Creative Commons Attribution (CC-BY)** or a similar license.
            """
        )

    app.load(
        check_auth_on_load,
        outputs=[
            repo_url, shared_input, send_first,
            feedback, submit_feedback_btn,
            hint_markdown, login_button,
            oauth_token,
        ],
    )

    app.launch()
