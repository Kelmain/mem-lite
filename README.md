# claude-mem-lite

Local memory for Claude Code sessions. Captures tool outputs, compresses them into structured observations via Claude Haiku, and injects relevant context at the start of each session.

## Quick Start

```bash
# Install
uv add claude-mem-lite

# Install hooks into Claude Code
claude-mem install

# Start the background worker
claude-mem-worker start

# Check system status
claude-mem status
```

## How It Works

1. **Capture**: Git hooks intercept Claude Code tool outputs and queue them.
2. **Compress**: A background worker compresses raw outputs into structured observations using Claude Haiku.
3. **Search**: Observations are embedded (Qwen3-Embedding-0.6B) and indexed in LanceDB for hybrid vector + full-text search.
4. **Inject**: At session start, relevant context (session history, function map, learnings, observations) is injected into the system prompt within a 2000-token budget.

## Commands

| Command | Description |
|---------|-------------|
| `claude-mem report` | Activity report (sessions, observations, function changes) |
| `claude-mem search <query>` | Search observations via worker or FTS fallback |
| `claude-mem mermaid [file]` | Generate Mermaid call graph diagrams |
| `claude-mem status` | System health check (database, worker, FTS) |
| `claude-mem eval compression` | Evaluate compression quality |
| `claude-mem eval health` | System health dashboard |
| `claude-mem compress --pending` | Compress pending items offline (no worker needed) |
| `claude-mem prune` | Clean up old data and reclaim disk space |

## Configuration

All data is stored in `~/.claude-mem/`:

- `claude-mem.db` -- SQLite database (WAL mode)
- `lance/` -- LanceDB vector store
- `worker.sock` -- Unix domain socket for worker communication
- `logs/` -- Worker logs

## Cost

Typical session cost is approximately $0.50-0.60, primarily from Claude Haiku compression calls.

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy .
```

## License

MIT
