# Security Policy

## Supported versions

Flexilearn is pre-1.0 and developed on `main`. Security fixes are applied to the latest
release only.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/DoubleH7/flexilearn/security/advisories/new)
(the *Security* tab → *Report a vulnerability*). Please include a description of the
issue, the steps to reproduce it, and the affected version or commit.

You can expect an initial response within a couple of weeks. This is a single-maintainer
research project, not a commercial product — please size your expectations accordingly.

## Things worth knowing before you report

Some behaviors look alarming but are intended, documented properties of a training
framework:

* **Configs execute code.** A `config.yml` names registry keys that map to Python classes,
  and `discover_definitions()` imports every module under `definitions/` at startup.
  Running a config from an untrusted source is equivalent to running untrusted code. Treat
  configs and `definitions/` directories as you would any other source file.
* **Checkpoints are unpickled.** Checkpoint loading uses
  `torch.load(..., weights_only=False)` because a checkpoint carries optimizer state, RNG
  state and task counters, not just tensors. Only resume from checkpoints you produced or
  otherwise trust.
* **`.env` is loaded at startup.** `flexilearn/entry.py` calls `load_dotenv()`, so a `.env`
  in the working directory reaches the process environment. `.env` is gitignored; see
  `.env.example`.
* **Tracking can send data off-machine.** The default `tracking_uri` is a local `./mlruns`
  directory. If you point it at a remote server, run metadata — including `git_branch` and
  the `uv.lock` artifact when `log_git_info` is true — goes to that server.

Reports about any of the above are still welcome if you have found a way to make them
behave *worse* than described here.
