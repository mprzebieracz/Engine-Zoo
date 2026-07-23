//! Standard Algebraic Notation and reusable PGN movetext formatting.

use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Square};
use engine_core::notation::GameNotation;
use std::fmt::Write;
use std::str::FromStr;

use super::position::ChessPosition;

#[derive(Clone, Copy, Debug, Default)]
pub struct ChessUciNotation;

impl GameNotation<ChessPosition> for ChessUciNotation {
    fn parse_move(&self, state: &ChessPosition, text: &str) -> Option<ChessMove> {
        let mv = ChessMove::from_str(text.trim()).ok()?;
        state.board.legal(mv).then_some(mv)
    }

    fn format_move(&self, _state: &ChessPosition, mv: ChessMove) -> String {
        mv.to_string()
    }
}

/// Formats a legal move as SAN. Invalid moves fall back to UCI notation.
pub fn san(board: &Board, mv: ChessMove) -> String {
    if !board.legal(mv) {
        return mv.to_string();
    }
    let Some(piece) = board.piece_on(mv.get_source())
    else {
        return mv.to_string();
    };
    if piece == Piece::King {
        match (
            mv.get_source().get_file().to_index(),
            mv.get_dest().get_file().to_index(),
        ) {
            (4, 6) => return with_check_suffix(board, mv, "O-O".into()),
            (4, 2) => return with_check_suffix(board, mv, "O-O-O".into()),
            _ => {}
        }
    }

    let capture = board.piece_on(mv.get_dest()).is_some()
        || (piece == Piece::Pawn && board.en_passant() == Some(mv.get_dest()));
    let mut text = String::with_capacity(8);
    if piece == Piece::Pawn {
        if capture {
            text.push(file_char(mv.get_source()));
        }
    }
    else {
        text.push(piece_char(piece));
        push_disambiguation(&mut text, board, piece, mv);
    }
    if capture {
        text.push('x');
    }
    write!(text, "{}", mv.get_dest()).expect("writing to String cannot fail");
    if let Some(promotion) = mv.get_promotion() {
        text.push('=');
        text.push(piece_char(promotion));
    }
    with_check_suffix(board, mv, text)
}

/// Formats SAN moves as PGN movetext, including move numbers and result.
pub fn movetext(moves: &[String], starting_ply: u16, result: &str) -> String {
    let mut out = String::new();
    for (offset, san) in moves.iter().enumerate() {
        let ply = usize::from(starting_ply) + offset;
        if !out.is_empty() {
            out.push(' ');
        }
        if ply % 2 == 0 {
            write!(out, "{}. {san}", ply / 2 + 1).unwrap();
        }
        else if offset == 0 {
            write!(out, "{}... {san}", ply / 2 + 1).unwrap();
        }
        else {
            out.push_str(san);
        }
    }
    if !result.is_empty() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(result);
    }
    out
}

/// Builds a PGN document from tag pairs and already-formatted movetext.
pub fn pgn(headers: &[(&str, &str)], movetext: &str) -> String {
    let mut out = String::new();
    for &(name, value) in headers {
        writeln!(out, "[{name} \"{}\"]", value.replace('"', "\\\"")).unwrap();
    }
    if !headers.is_empty() {
        out.push('\n');
    }
    out.push_str(movetext);
    out.push('\n');
    out
}

fn push_disambiguation(out: &mut String, board: &Board, piece: Piece, mv: ChessMove) {
    let source = mv.get_source();
    let mut conflict = false;
    let mut same_file = false;
    let mut same_rank = false;
    for other in MoveGen::new_legal(board) {
        if other != mv
            && other.get_dest() == mv.get_dest()
            && board.piece_on(other.get_source()) == Some(piece)
        {
            conflict = true;
            same_file |= other.get_source().get_file() == source.get_file();
            same_rank |= other.get_source().get_rank() == source.get_rank();
        }
    }
    if conflict {
        if !same_file {
            out.push(file_char(source));
        }
        else if !same_rank {
            out.push(rank_char(source));
        }
        else {
            out.push(file_char(source));
            out.push(rank_char(source));
        }
    }
}

fn with_check_suffix(board: &Board, mv: ChessMove, mut text: String) -> String {
    let next = board.make_move_new(mv);
    if next.status() == BoardStatus::Checkmate {
        text.push('#');
    }
    else if next.checkers().popcnt() > 0 {
        text.push('+');
    }
    text
}

fn piece_char(piece: Piece) -> char {
    match piece {
        Piece::Knight => 'N',
        Piece::Bishop => 'B',
        Piece::Rook => 'R',
        Piece::Queen => 'Q',
        Piece::King => 'K',
        Piece::Pawn => unreachable!("pawn has no SAN piece letter"),
    }
}

fn file_char(square: Square) -> char {
    (b'a' + square.get_file().to_index() as u8) as char
}

fn rank_char(square: Square) -> char {
    (b'1' + square.get_rank().to_index() as u8) as char
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn uci_notation_parses_and_formats_legal_moves() {
        let position = ChessPosition::default();
        let mv = ChessUciNotation.parse_move(&position, " e2e4 ").unwrap();
        assert_eq!(ChessUciNotation.format_move(&position, mv), "e2e4");
        assert_eq!(ChessUciNotation.parse_move(&position, "e2e5"), None);
    }

    fn san_at(fen: &str, mv: &str) -> String {
        let board = Board::from_str(fen).unwrap();
        san(&board, ChessMove::from_str(mv).unwrap())
    }

    #[test]
    fn formats_castles_captures_promotions_and_checks() {
        assert_eq!(
            san_at("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            "O-O"
        );
        assert_eq!(san_at("4k3/8/8/8/8/8/4p3/4K3 w - - 0 1", "e1e2"), "Kxe2");
        assert_eq!(san_at("k7/6P1/8/8/8/8/8/K7 w - - 0 1", "g7g8q"), "g8=Q+");
        assert_eq!(
            san_at(
                "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "h5f7"
            ),
            "Qxf7#"
        );
    }

    #[test]
    fn disambiguates_by_file_rank_and_both() {
        assert_eq!(san_at("4k3/8/8/8/8/8/3N3N/4K3 w - - 0 1", "d2f3"), "Ndf3");
        assert_eq!(san_at("4k3/8/8/8/8/3N4/8/3NK3 w - - 0 1", "d1f2"), "N1f2");
        assert_eq!(san_at("4k3/8/8/N7/8/8/8/N1N1K3 w - - 0 1", "a1b3"), "Na1b3");
    }

    #[test]
    fn formats_move_numbers_results_and_document() {
        let moves = ["e4".into(), "e5".into(), "Nf3".into()];
        assert_eq!(movetext(&moves, 0, "1-0"), "1. e4 e5 2. Nf3 1-0");
        assert_eq!(movetext(&["e5".into()], 1, "*"), "1... e5 *");
        assert_eq!(
            pgn(&[("Event", "test"), ("Result", "1-0")], "1. e4 1-0"),
            "[Event \"test\"]\n[Result \"1-0\"]\n\n1. e4 1-0\n"
        );
    }
}
