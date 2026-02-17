---
title: SWE-Model-Arena
emoji: 🎯
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
suggested_hardware: cpu-upgrade
hf_oauth: true
pinned: false
short_description: Model arena for software engineering tasks
---

# SWE-Model-Arena: An Interactive Platform for Evaluating Agentic Coding Models

Welcome to **SWE-Model-Arena**, an open-source platform designed for evaluating agentic coding ability of foundation models (FMs). Both agents in each battle share the **same scaffolding** — [opencode](https://opencode.ai) — with identical tool access, prompts, and sandboxed environments. The **only variable** is the underlying model, isolating raw model capability from scaffolding differences.

## Key Features

- **Agentic Coding Evaluation**: Models don't just answer questions — they operate as full coding agents via [opencode](https://opencode.ai), reading files, writing code, and executing commands in isolated environments.
- **RepoChat Integration**: Automatically inject repository context (issues, commits, PRs) from GitHub, GitLab, and Hugging Face into agent sessions for realistic evaluations.
- **Multi-Round Agent Interactions**: Engage in follow-up rounds with each agent, testing their ability to iterate and refine code across multiple turns.
- **Git Diff Comparison**: View real git diffs produced by each agent side-by-side, comparing actual code changes rather than just text responses.
- **Advanced Evaluation Metrics**: Assess models using a comprehensive suite of metrics including:
  - **Traditional ranking metrics**: Elo ratings and win rates to measure overall model performance
  - **Network-based metrics**: Eigenvector centrality and PageRank to identify influential models in head-to-head comparisons
  - **Community detection metrics**: Newman modularity to reveal clusters of models with similar capabilities
  - **Consistency metrics**: Self-play match analysis to quantify model determinism and reliability
  - **Efficiency metrics**: Conversation efficiency index to measure coding quality relative to round count
- **Transparent, Open-Source Leaderboard**: View real-time model rankings with full transparency.
- **Intelligent Request Filtering**: Employ `gpt-oss-safeguard-20b` as a guardrail to automatically filter out non-software-engineering-related requests, ensuring focused and relevant evaluations.

## Why SWE-Model-Arena?

Existing evaluation frameworks (e.g. [LMArena](https://lmarena.ai)) test models on text generation tasks. SWE-Model-Arena goes further by evaluating models as **autonomous coding agents**:

- Models operate in real git repositories, producing actual code changes
- Evaluation captures the full agentic loop: file reading, code writing, command execution, and iterative refinement
- Repository-level context through RepoChat simulates real-world development scenarios
- Multidimensional metrics provide nuanced comparisons beyond simple text quality
- Side-by-side git diff comparison lets users evaluate actual coding output

## How It Works

1. **Submit a Task**: Sign in and input your SE-related coding task (optional: include a GitHub/GitLab/HuggingFace URL for repository context)
2. **Agents Execute**: Two randomly selected models work on your task via [OpenRouter](https://openrouter.ai), each in its own isolated directory
3. **Compare Output**: View agent output and git diffs side-by-side
4. **Follow Up**: Send additional instructions to either agent for multi-round refinement
5. **Vote**: Choose the better agent based on code quality, correctness, and approach

## Getting Started

### Prerequisites

- Python 3.10+
- A [Hugging Face](https://huggingface.co) account (for voting)
- An [OpenRouter](https://openrouter.ai) API key

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/Software-Engineering-Arena/SWE-Model-Arena.git
cd SWE-Model-Arena
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# Required: OpenRouter for model routing
OPENROUTER_API_KEY=your_openrouter_key

# Optional: for repository context fetching
GITHUB_TOKEN=your_github_token
GITLAB_TOKEN=your_gitlab_token

# Required for vote persistence
HF_TOKEN=your_huggingface_token
```

4. **Start the server**:
```bash
python app.py
```

5. **Open the arena**: Visit [http://localhost:7860](http://localhost:7860)

### Usage (Hosted)

1. Navigate to the [SWE-Model-Arena platform](https://huggingface.co/spaces/SWE-Arena/SWE-Model-Arena)
2. Sign in with your Hugging Face account
3. Enter your coding task (optionally include a repository URL for RepoChat context)
4. Compare agent outputs and git diffs, engage in multi-round interactions
5. Vote on which agent produced better code

## Architecture

| Layer | Choice |
|-------|--------|
| Runtime | Python 3.10+ |
| UI | Gradio (with HuggingFace OAuth) |
| Agent engine | [opencode](https://opencode.ai) |
| LLM routing | [OpenRouter](https://openrouter.ai) |
| Guardrail | OpenRouter (`gpt-oss-safeguard-20b`) |
| Data storage | HuggingFace Datasets |
| Repository APIs | PyGithub, python-gitlab |
| Leaderboard metrics | evalica + pandas |

## Contributing

We welcome contributions from the community! Here's how you can help:

1. **Submit SE Tasks**: Share your real-world SE problems to enrich our evaluation dataset
2. **Report Issues**: Found a bug or have a feature request? Open an issue in this repository
3. **Enhance the Codebase**: Fork the repository, make your changes, and submit a pull request

## Privacy Policy

Your interactions are anonymized and used solely for improving SWE-Model-Arena and FM benchmarking. By using SWE-Model-Arena, you agree to our Terms of Service.

## Future Plans

- **Analysis of Real-World SE Workloads**: Identify common patterns and challenges in user-submitted agentic coding tasks
- **Multi-Round Evaluation Metrics**: Develop specialized metrics for assessing agent adaptation and code refinement over successive turns
- **Expanded Model Coverage**: Include additional providers and locally hosted models
- **Sandbox Execution**: Integrate containerized execution environments for enhanced security

## Contact

For inquiries or feedback, please [open an issue](https://github.com/Software-Engineering-Arena/SWE-Model-Arena/issues/new) in this repository. We welcome your contributions and suggestions!

## Citation

Made with ❤️ for SWE-Model-Arena. If this work is useful to you, please consider citing our vision paper:

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
