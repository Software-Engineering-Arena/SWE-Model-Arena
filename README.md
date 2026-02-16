---
title: SWE Model Arena
emoji: 🎯
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
hf_oauth: true
pinned: false
short_description: Model arena for software engineering tasks
---

# SWE-Assistant-Arena: Interactive Platform for Evaluating Coding Assistants

Welcome to **SWE-Assistant-Arena**, a minimalist open-source platform for evaluating **coding assistants** on real software engineering tasks through head-to-head battles.

## 🤖 Supported Assistants

The arena currently supports **5 coding assistants** via [VibeKit](https://github.com/superassistant-ai/vibekit):

- **Claude** - Anthropic's Claude Code CLI
- **Gemini** - Google's Gemini CLI
- **Codex** - OpenAI's Codex CLI
- **OpenCode** - OpenCode CLI
- **Grok** - xAI's Grok CLI

## ✨ Key Features

- **Real Code Execution**: Assistants work in sandboxed environments via VibeKit
- **Random Head-to-Head Battles**: Two random assistants compete on each task
- **Minimalist Design**: Clean, focused interface with no unnecessary features
- **Elo Rating System**: Track assistant performance with Elo rankings
- **Real-time Voting**: Vote on which assistant produces better code
- **Transparent Leaderboard**: See wins, losses, and Elo ratings for all assistants

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ installed
- API keys for the assistants you want to use
- [E2B](https://e2b.dev/) account and API key (for sandboxed code execution)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Software-Engineering-Arena/SWE-Assistant-Arena.git
cd SWE-Assistant-Arena
```

2. **Install dependencies**:
```bash
npm install
```

3. **Configure environment variables**:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# Required: E2B for sandboxed execution
E2B_API_KEY=your_e2b_api_key

# Add keys for assistants you want to use
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
GROK_API_KEY=your_grok_key
```

4. **Start the server**:
```bash
npm start
```

5. **Open the arena**:
Visit [http://localhost:7860](http://localhost:7860)

## 🚀 Deploy to Hugging Face Spaces

This app is ready to deploy on Hugging Face Spaces:

1. **Create a new Space** on [Hugging Face](https://huggingface.co/spaces)
2. **Select Docker SDK** when creating the Space
3. **Push this repository** to your Space
4. **Add Secrets** in Space settings:
   - `E2B_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`
   - `GROK_API_KEY`
5. Your Space will automatically build and deploy!

## 🎮 How to Use

1. **Enter a coding task** in the prompt field (e.g., "Create a REST API endpoint for user authentication")
2. **Click "Start Battle"** to execute two random assistants
3. **Review the outputs** from both assistants, including execution time and logs
4. **Vote** for which assistant produced better code
5. **Check the leaderboard** to see rankings

## 📊 Leaderboard Metrics

- **Elo Rating**: Standard Elo rating system (starting at 1000)
- **Wins/Losses/Ties**: Total match results
- **Total Matches**: Number of battles participated in

## 🛠️ Architecture

The arena is built with:

- **Backend**: Node.js + Express
- **Assistant Execution**: VibeKit SDK with E2B sandboxes
- **Frontend**: Vanilla JavaScript with minimal dependencies
- **Storage**: In-memory (for simplicity)

## 🔒 Security

- All code execution happens in isolated E2B sandboxes
- No code runs on your local machine
- VibeKit provides built-in security and secret redaction

## 🤝 Contributing

Contributions are welcome! To add features:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 API Endpoints

- `GET /api/assistants/random` - Get two random assistants
- `POST /api/battle/start` - Start a battle between two assistants
- `POST /api/vote` - Submit a vote for a battle
- `GET /api/leaderboard` - Get current leaderboard

## 🚧 Roadmap

- [ ] Persistent storage (database integration)
- [ ] User authentication
- [ ] Battle history viewing
- [ ] Custom assistant configurations
- [ ] More detailed execution metrics
- [ ] Docker deployment setup
- [ ] Support for more VibeKit assistants

## 📚 Resources

- [VibeKit Documentation](https://docs.vibekit.sh/)
- [E2B Documentation](https://e2b.dev/docs)
- [Original Paper](https://arxiv.org/abs/2502.01860)

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [VibeKit](https://github.com/superassistant-ai/vibekit) for assistant execution framework
- [E2B](https://e2b.dev/) for sandboxed code execution
- Original SWE-Arena concept and design

## 📧 Contact

For inquiries or feedback, please [open an issue](https://github.com/Software-Engineering-Arena/SWE-Assistant-Arena/issues) in this repository.

## 📖 Citation

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
