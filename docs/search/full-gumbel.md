# Full Gumbel MCTS

Full Gumbel search is implemented in `crates/search/src/mcts/gumbel.rs`. It samples Gumbel noise at the root, considers a bounded candidate set, and allocates simulations through sequential halving. Completed-Q transforms make incomplete child values comparable during that schedule.

The final policy target is the search-improved distribution defined by root logits plus transformed completed-Q, not an ordinary PUCT visit-count target. `GumbelConfig` owns algorithm constants such as candidate count and Q transform; `SearchBudget::Gumbel` changes only simulations and candidate count for a move.

Checked-in schedule fixtures and golden tests protect the allocation sequence. Parallelism is intentionally conservative: correctness of root candidate accounting is more valuable than speculative concurrent scheduling.

