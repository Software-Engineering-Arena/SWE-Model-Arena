---
title: SWE-Model-Arena
emoji: 🎯
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
suggested_hardware: cpu-upgrade
hf_oauth: true
pinned: false
short_description: Model arena for software engineering tasks
---

# SWE-Model-Arena

An open-source platform for evaluating agentic coding models head-to-head. Both sides share the **same scaffolding** ([opencode](https://opencode.ai)) with identical tools, prompts, and sandboxed environments — the **only variable** is the underlying tool-calling model.

**[Try it on Hugging Face Spaces](https://huggingface.co/spaces/SWE-Arena/SWE-Model-Arena)**

## Key Capabilities

- **Agentic evaluation** — models read files, write code, and execute commands in real git repos via [opencode](https://opencode.ai), not just generate text
- **RepoChat** — auto-injects repo context (issues, commits, PRs) from GitHub / GitLab / Hugging Face
- **Multi-round + git diff comparison** — follow up across turns and compare actual diffs side-by-side
- **Rich leaderboard** — Elo, PageRank, modularity clustering, self-play consistency, and efficiency metrics

## How It Works

1. **Submit a task** — sign in, enter a coding task (optionally include a repo URL for RepoChat context)
2. **Models execute** — two randomly selected tool-calling models work independently via [OpenRouter](https://openrouter.ai)
3. **Compare** — view outputs and git diffs side-by-side; send follow-ups for multi-round refinement
4. **Vote** — pick the better model based on code quality, correctness, and approach

## Contributing

Submit tasks, report bugs, or request features via [GitHub Issues](https://github.com/Software-Engineering-Arena/SWE-Model-Arena/issues/new).

## Citation

```bibtex
@inproceedings{zhao2025se,
  title={SE Arena: An Interactive Platform for Evaluating Foundation Models in Software Engineering},
  author={Zhao, Zhimin},
  booktitle={2025 IEEE/ACM Second International Conference on AI Foundation Models and Software Engineering (Forge)},
  pages={78--81},
  year={2025},
  organization={IEEE}
}
```
