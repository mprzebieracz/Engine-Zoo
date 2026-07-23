# PUCT search

The PUCT implementation is in `crates/search/src/mcts/{puct,core,traversal,evaluation}.rs`. Node values are always from the player-to-move perspective. Backing a value up one edge flips its sign.

Selection combines a child value estimate with its prior and visit count. FPU supplies an explicit value for unvisited children; virtual visits/loss make an in-flight leaf less attractive to a concurrent local leaf batch. Root Dirichlet noise is a self-play option, not an evaluation option.

`CommonSearchConfig.leaf_batch_size` is an execution setting. `SearchBudget::Puct` contains only simulations, because a per-move budget must not pretend to change the fixed batching scheduler. Policy targets are normalized root visit counts over legal native moves.

Evaluator output is validated before it enters a node: cardinality, legal-logit count, finite logits, and finite bounded value are mandatory. Cache diagnostics distinguish backend calls, hits, and misses.

