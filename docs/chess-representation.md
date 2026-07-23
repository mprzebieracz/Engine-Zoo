# Chess representations

The canonical model uses `ChessAzRepresentation<HISTORY>` and a 73-plane policy per origin square (4,672 actions). Its encoded state includes bounded board history and chess-specific features, so its cache key includes every encoded feature: clocks, repetition, castling/en-passant state, side to move, and history.

The classic model uses `ChessClassicRepresentation`: 19 input planes and a 64×64×5 = 20,480 action layout with a scalar value head. It remains a first-class `ModelSpec::chess_classic` option so existing checkpoints remain usable. Its source file names preserve historical parameter naming; that is compatibility, not a signal that the model is unsupported.

Rules never consume either action layout. Native `chess::ChessMove` is converted only at the representation/inference/replay boundary.

