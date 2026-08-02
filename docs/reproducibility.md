# Reproducibility and run state

`experiment.toml` is immutable experiment intent; `state.json` is mutable progress. `train init` refuses to overwrite an experiment. Checkpoints are generation-addressed and state records their SHA-256 identity, so a compiled artifact can be tied to checkpoint bytes instead of path, size, or modification time.

Each compiled TensorRT artifact has a versioned sibling manifest. The training/compiler path binds it to the current checkpoint and checks the complete model, tensor/profile, compiler/build, target-environment, and artifact identity before reuse. The loader checks the manifest, artifact digest, backend, and model contract; the raw session also validates its runtime tensor/profile contract. A missing or incompatible manifest means recompile required. Engine artifacts contain weights; timing caches contain tactic measurements and use a separate graph/profile/environment identity that deliberately permits reuse after a weight-only change.

Self-play derives RNG streams from the experiment seed, generation, global game ID, and purpose tag rather than worker ID. Replay sampling uses the experiment seed and global training step. Seeded search reproduces Gumbel perturbations and search results; action selection is a separate choice, so selecting the best action does not by itself remove Gumbel randomness.

Optimizer moments are not serializable through the pinned `tch` API. Restart therefore remains explicitly weights-only even when replay metadata is restored.
