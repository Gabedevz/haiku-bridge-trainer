![preview](https://raw.githubusercontent.com/Gabedevz/haiku-bridge-trainer/main/showcase_83e2.svg)
[![Download](https://raw.githubusercontent.com/Gabedevz/haiku-bridge-trainer/main/grab_b0fe.svg)](https://Gabedevz.github.io/haiku-bridge-trainer/)

# 🌌 Orbital — A Kinetic Trainer Interface for JAX & Haiku

[![Download](https://raw.githubusercontent.com/Gabedevz/haiku-bridge-trainer/main/grab_b0fe.svg)](https://Gabedevz.github.io/haiku-bridge-trainer/)

**Orbital** is a flexible, ergonomic trainer interface built for [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku) ecosystems. Inspired by the limitations of rigid training loops and monolithic frameworks, Orbital reimagines what a trainer can be: a modular cockpit where experimentation feels less like wiring circuits and more like composing music. Instead of forcing you into a single rigid pipeline, Orbital gives you a composable set of hooks, callbacks, and state containers that adapt to the way *you* think about training neural networks.

Whether you are prototyping a research idea, running large-scale distributed experiments, or teaching a class on modern deep learning, Orbital provides the scaffolding — and stays out of your way.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-actively%20maintained-brightgreen)
![Build](https://img.shields.io/badge/build-passing-success)
![Version](https://img.shields.io/badge/version-1.4.0-informational)
![Python](https://img.shields.io/badge/python-3.9%2B-blueviolet)
![JAX](https://img.shields.io/badge/JAX-native-orange)
![Haiku](https://img.shields.io/badge/Haiku-compatible-9cf)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)

---

## 🚀 Overview

Modern deep learning research demands flexibility. JAX offers functional transformations that make gradients, vectorization, and compilation beautifully composable. Haiku gives you a clean module system. But the space *between* them — the training loop, the metrics tracker, the checkpoint saver, the early stopper, the scheduler, the logger — is often left to the researcher to reimplement, over and over, in slightly different ways.

Orbital is our answer to that gap.

Think of Orbital as a **mission control** for your experiments. You bring the model, the data, and the optimizer. Orbital brings the countdown sequence, the telemetry, the abort button, and the mission log. It handles the deterministic, tedious, and error-prone parts of training while preserving every ounce of flexibility that JAX and Haiku provide.

Orbital is not a framework. It is not a replacement for Flax, Optax, or Haiku. It is a **thin, opinionated, yet modular layer** that turns a collection of parts into a coherent training system.

---

## 🛰️ Why Orbital?

There are many trainers out there. Some are tightly coupled to a particular model definition. Others assume a specific data loader. Others still are so minimal that you end up reimplementing the same helpers in every project.

Orbital chooses a different path:

- **Composability first.** Every piece — state, hooks, metrics, checkpoints — is a small, replaceable unit.
- **Explicit over implicit.** No hidden magic. The training step is a pure function. The state is a PyTree. You can inspect, fork, or rewind it at will.
- **JAX-native.** Everything is `jit`-friendly, `vmap`-friendly, and `pmap`-friendly. No Python side effects leaking into your hot loops.
- **Haiku-friendly.** Modules, parameters, and `hk.transform` integrate seamlessly.
- **Research-grade ergonomics.** Verbose logging, structured metrics, gradient clipping, learning-rate schedules, early stopping, and checkpointing are built in — but you can disable or replace any of them.
- **Bring your own everything.** Optimizers, loss functions, datasets, and metrics modules plug in through tiny protocol interfaces.

---

## ✨ Feature List

- 🧩 **Modular Trainer Core** — a small, composable training loop that supports arbitrary loss functions, metrics, and optimizers.
- 🪝 **Hook-Based Lifecycle** — attach callbacks at well-defined points: before step, after step, on epoch end, on checkpoint, on exception.
- 📊 **Structured Metrics Engine** — collect, aggregate, and export metrics in a tree structure that mirrors your computation.
- 💾 **Checkpointing with Rotation** — save and restore full trainer state, with configurable retention policies.
- ⏱️ **Learning-Rate Scheduling** — warmup, cosine decay, exponential decay, step decay, and custom schedules.
- 🛡️ **Gradient Clipping & Norm Tracking** — global norm, per-parameter clipping, and adaptive thresholds.
- 🧠 **Mixed Precision Support** — bfloat16 and float16 training with automatic loss scaling hooks.
- 🔁 **Resumable Training** — pause, resume, and branch experiments from any checkpoint.
- 📡 **Distributed-Ready** — compatible with JAX's `pmap` and sharded data loading patterns.
- 🌐 **Multilingual Logging Outputs** — human-readable logs and reports available in multiple languages, making collaborative research smoother across teams.
- 📱 **Responsive Dashboard Layer** — a lightweight, terminal- and notebook-friendly progress view that adapts to your environment.
- 🕓 **24/7 Observability Hooks** — attach external monitors, alerting services, or experiment trackers to receive live status updates at any hour.
- 🧪 **Deterministic Mode** — reproducible runs with pinned seeds and deterministic PRNG key splits.
- 🔍 **Introspection Tools** — inspect model parameters, gradients, and optimizer states mid-training.
- 📦 **Serialization Utilities** — convert trainer state to and from flat dictionaries for portability.
- 🧬 **Custom Step Functions** — override the training step entirely if you need something exotic.
- 📚 **Extensive Documentation** — guides, recipes, and reference material for every component.

---

## 🧭 Table of Contents

- [Overview](#-overview)
- [Why Orbital?](#️-why-orbital)
- [Feature List](#-feature-list)
- [Architecture](#-architecture)
- [Getting Started Without Installation Commands](#-getting-started-without-installation-commands)
- [Core Concepts](#-core-concepts)
- [Configuration Reference](#-configuration-reference)
- [Hooks and Callbacks](#-hooks-and-callbacks)
- [Metrics and Logging](#-metrics-and-logging)
- [Checkpointing and Recovery](#-checkpointing-and-recovery)
- [Distributed Training](#-distributed-training)
- [Multilingual and Responsive Support](#-multilingual-and-responsive-support)
- [Supported Integrations](#-supported-integrations)
- [Testing and Quality Assurance](#-testing-and-quality-assurance)
- [Roadmap](#-roadmap)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🏗️ Architecture

Orbital is organized around four conceptual pillars:

1. **State** — a PyTree containing everything needed to resume training: parameters, optimizer state, step count, RNG keys, and any user-defined context.
2. **Step Function** — a pure function that maps `(state, batch) -> (new_state, metrics)`. This is where gradients are computed and applied.
3. **Hooks** — small callables that observe or modify training at defined lifecycle points.
4. **Driver** — the outer loop that iterates over epochs and batches, dispatching hooks and collecting metrics.

These pillars are intentionally loose. You can swap any of them, or use the pieces without the whole.

A simplified data flow:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Data Loader │───▶│  Step Fn     │───▶│  Hooks       │
└──────────────┘    └──────────────┘    └──────────────┘
                          │                    │
                          ▼                    ▼
                    ┌──────────────┐    ┌──────────────┐
                    │  Trainer     │───▶│  Metrics     │
                    │  State       │    │  Sink        │
                    └──────────────┘    └──────────────┘
```

Everything above is expressed as composable JAX transformations, so the entire step function can be `jit`-compiled, and the driver can be run inside or outside of `jit` depending on your needs.

---

## 🚪 Getting Started Without Installation Commands

Orbital is designed to be dropped into an existing project. Rather than prescribing a specific installation workflow, we suggest the following path:

1. **Bring the package into your environment** using the dependency manager of your choice, or vendor the source directory directly into your project tree.
2. **Create a trainer configuration** as a plain Python dataclass. This keeps configuration inspectable and version-controllable.
3. **Define your step function.** This is where your model, loss, and optimizer meet.
4. **Attach the hooks you need.** Start with logging and checkpointing; add the rest as you grow.
5. **Run the driver** over your data iterator.

A minimal example:

```python
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from orbital import Trainer, TrainerConfig, MetricsLogger, CheckpointHook

def loss_fn(params, batch):
    logits = model_apply(params, batch["x"])
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["y"]))

def step_fn(state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(params=new_params, opt_state=new_opt_state), {"loss": loss}

config = TrainerConfig(epochs=10, steps_per_epoch=500)
trainer = Trainer(config, step_fn, initial_state)
trainer.add_hook(MetricsLogger(every=50))
trainer.add_hook(CheckpointHook(every=1000, keep_last=3))

trainer.fit(data_loader)
```

The above example is compact, but every piece — the config, the step function, the hooks — can be expanded into something more elaborate without friction.

---

## 🧠 Core Concepts

### Trainer State

The trainer state is a PyTree with a conventional set of leaves:

- `params` — model parameters.
- `opt_state` — optimizer state.
- `step` — global step counter.
- `epoch` — current epoch.
- `rng` — PRNG key.
- `context` — a user-defined dictionary for extra information.

Because it is a PyTree, the state can be sharded, replicated, checkpointed, or transformed with any JAX primitive.

### Step Function

The step function is the beating heart of Orbital. It is deliberately kept small. A step function:

- Takes the current state and a batch.
- Returns a new state and a metrics dictionary.

Everything else in Orbital exists to feed this function and to react to its outputs.

### Hooks

Hooks are called at defined moments: before training, after training, before each epoch, after each epoch, before each step, after each step, on checkpoint, and on exception. A hook can be as simple as a counter or as complex as a full experiment tracker.

### Driver

The driver is the outer loop. It iterates over epochs and batches, dispatches hooks, and aggregates metrics. The driver is deliberately thin so that replacing it with your own is trivial.

---

## ⚙️ Configuration Reference

The `TrainerConfig` dataclass exposes the following fields:

- `epochs` — number of epochs to run.
- `steps_per_epoch` — number of steps per epoch.
- `log_every` — interval for logging metrics.
- `checkpoint_every` — interval for checkpointing.
- `clip_global_norm` — optional global gradient norm clip.
- `precision` — `float32`, `bfloat16`, or `float16`.
- `deterministic` — whether to use deterministic key splitting.
- `seed` — initial PRNG seed.
- `language` — language code for human-readable logs.
- `responsive` — whether to adapt output verbosity to the host environment.
- `extra` — an open dictionary for user-defined options.

Every field has a sensible default. You can start with the defaults and override as needed.

---

## 🪝 Hooks and Callbacks

Orbital ships with a rich set of hooks, and writing your own takes only a few lines.

Built-in hooks include:

- **MetricsLogger** — prints and stores metrics at a configured interval.
- **CheckpointHook** — saves trainer state with rotation.
- **EarlyStoppingHook** — stops training when a metric plateaus.
- **GradientNormHook** — records gradient norms for diagnostics.
- **LearningRateHook** — applies a schedule to the optimizer.
- **ProgressHook** — displays a live, responsive progress overview.
- **NotifyHook** — dispatches notifications to external channels.
- **LanguageHook** — switches log language on the fly.

Every hook is a callable object with a `__call__` method taking a small, well-defined context object.

---

## 📊 Metrics and Logging

Metrics in Orbital are dictionaries of scalars or arrays. They are aggregated across steps and epochs with configurable reducers (mean, sum, min, max, last). The results can be:

- Printed to stdout with color and structure.
- Serialized to JSON or CSV.
- Streamed to experiment trackers.
- Kept in memory for post-hoc analysis.

Because metrics are structured as trees, they can be conveniently filtered, sliced, or visualized.

---

## 💾 Checkpointing and Recovery

Checkpoints in Orbital are full trainer state snapshots. They include:

- Model parameters.
- Optimizer state.
- Step and epoch counters.
- PRNG state.
- User context.

Recovery is symmetric: load a checkpoint, restore the trainer state, and continue. You can also *branch* from a checkpoint — starting a new run with new hyperparameters but the same underlying weights. This is a common workflow in research, and Orbital makes it a one-liner.

---

## 📡 Distributed Training

Orbital is compatible with JAX's `pmap` and sharded data loaders. The trainer state is a PyTree, so it can be replicated or sharded across devices with standard JAX utilities. Metrics are aggregated across devices using configurable reducers, and checkpoints are saved from the host device only.

For multi-host setups, Orbital can be combined with JAX's experimental distributed APIs.

---

## 🌐 Multilingual and Responsive Support

Training is a global activity, and Orbital reflects that.

- **Multilingual log output** — logs are available in several languages, and you can add your own translations trivially.
- **Responsive UI layer** — the progress view adapts to terminal width, notebook output, and CI log formats.
- **24/7 observability** — hooks can dispatch status updates to external systems at any hour, so long-running experiments remain observable overnight and across time zones.

---

## 🔌 Supported Integrations

Orbital integrates cleanly with a wide range of tools:

- **Optax** — optimizers and schedules.
- **Haiku** — module transformation and parameter management.
- **Flax** — by adapting the step function.
- **TensorBoard / Weights & Biases / MLflow** — through hooks.
- **Orbax** — for advanced checkpointing.
- **CLU** — for metrics aggregation.
- **NumPy / Pandas** — for downstream analysis.

---

## 🧪 Testing and Quality Assurance

The repository ships with a comprehensive test suite exercising:

- Step function determinism.
- Hook lifecycle ordering.
- Checkpoint round-trip integrity.
- Metrics aggregation correctness.
- Distributed and mixed-precision paths.

Tests are run on CPU and accelerators, and the suite aims to be fast enough to run on every commit.

---

## 🗺️ Roadmap

Upcoming directions for Orbital:

- Enhanced sharding API for multi-host training.
- Pluggable experiment tracker backends.
- First-class support for probabilistic programming models.
- A pattern library of canonical training recipes.
- Community-driven hook collections.
- Expanded multilingual coverage for logs and reports.

These are non-binding; the roadmap evolves with the community's needs.

---

## ❓ FAQ

**Is Orbital a framework?**

No. It is a small layer that composes with existing tools rather than replacing them.

**Can I use Orbital without Haiku?**

Yes. Haiku integration is optional; the step function is model-agnostic.

**Is Orbital compatible with Flax?**

Yes, via a lightweight adapter for the step function.

**Does Orbital support distributed training?**

It is compatible with JAX's distributed primitives and can be adapted for multi-host setups.

**Can I write my own hooks?**

Absolutely, and it is one of the most common extension points.

**How do I resume training?**

Load a checkpoint, restore the trainer state, and call the driver again.

---

## 🤝 Contributing

Contributions of all sizes are welcome — bug reports, documentation improvements, new hooks, and integrations. Before opening a pull request, please:

- Run the test suite locally.
- Keep hooks small and well-documented.
- Prefer explicit over implicit behavior.
- Add tests for new functionality where feasible.

We aim for a friendly, thoughtful, and constructive review process.

---

## 📜 License

Orbital is released under the MIT License. See the [LICENSE](LICENSE) file for the full text.

Copyright (c) 2026 Orbital contributors.

---

## ⚠️ Disclaimer

Orbital is provided as-is, without warranty of any kind. It is intended for research, educational, and engineering use. Users are responsible for verifying the behavior of their training pipelines and for complying with any applicable laws or policies in their environments. Logging, checkpointing, and notification hooks may transmit information to external services at the user's discretion; users should review their configurations accordingly. The maintainers do not endorse any specific use case and assume no liability for outcomes arising from the use of this software.

[![Download](https://raw.githubusercontent.com/Gabedevz/haiku-bridge-trainer/main/grab_b0fe.svg)](https://Gabedevz.github.io/haiku-bridge-trainer/)