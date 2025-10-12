# src/puzzles_evaluator.py
# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import chess
import chess.pgn
import io
import numpy as np
import pandas as pd
import haiku as hk

from absl import logging

from searchless_chess.src import constants
from searchless_chess.src import tokenizer
from searchless_chess.src.engines import neural_engines

from searchless_chess.src import utils
from searchless_chess.src.engines import engine as engines_core

@dataclass
class PuzzlesEvalConfig:
  puzzles_path: str | None = None       # default resolves to repo_root/data/puzzles.csv
  num_puzzles: int = 512                # how many to evaluate each time
  batch_size: int = 64                  # for the predict_fn wrapper
  policy: str = "behavioral_cloning_param"  # used to choose engine
  num_return_buckets: int = 128

def _evaluate_puzzle_from_board(
    board: chess.Board,
    moves: Sequence[str],
    engine,
) -> bool:
  """Return True if engine solves the Lichess puzzle line.

  Lichess puzzles consider all mate-in-1 as correct if engine deviates but mates.
  See comment in original helper.
  """
  for move_idx, move in enumerate(moves):
    # Position is before opponent move; first move is applied before our choice.
    if move_idx % 2 == 1:
      predicted_move = engine.play(board=board).uci()
      if move != predicted_move:
        board.push(chess.Move.from_uci(predicted_move))
        return board.is_checkmate()
    board.push(chess.Move.from_uci(move))
  return True

# ADD:
class _ActionValuePlayAdapter:
  """Wraps an action-value engine to expose a .play(board) -> chess.Move."""
  def __init__(self, av_engine, return_bucket_values: np.ndarray):
    self._eng = av_engine
    self._rbv = np.asarray(return_bucket_values, dtype=np.float32)

  def play(self, board: chess.Board) -> chess.Move:
    # Analyse returns per-legal log-probs over return buckets (shape [N, R]).
    res = self._eng.analyse(board)
    rb_log_probs = res['log_probs']
    rb_probs = np.exp(rb_log_probs)              # [N, R]
    # Expected win prob per legal: [N]
    win_probs = rb_probs @ self._rbv
    legal_moves = engines_core.get_ordered_legal_moves(board)
    best_idx = int(np.argmax(win_probs))
    return legal_moves[best_idx]

def _engine_from_policy(policy: str, predict_fn, num_return_buckets: int):
  if policy == "behavioral_cloning_param":
    return neural_engines.ParamBCEngine(predict_fn=predict_fn)

  elif policy == "behavioral_cloning":
    return neural_engines.BCEngine(predict_fn=predict_fn)

  elif policy == "action_value_param":
    # Build param-based action-value engine, then wrap into a .play(...) adapter.
    _, rb_values = utils.get_uniform_buckets_edges_values(num_return_buckets)
    av_engine = neural_engines.ActionValueParamEngine(
        rb_values,
        predict_fn,
    )
    return _ActionValuePlayAdapter(av_engine, rb_values)

  elif policy == "action_value":
    # Original action-id-based action-value engine -> wrap into .play(...) adapter.
    _, rb_values = utils.get_uniform_buckets_edges_values(num_return_buckets)
    av_engine = neural_engines.ActionValueEngine(
        rb_values,
        predict_fn,
    )
    return _ActionValuePlayAdapter(av_engine, rb_values)

  else:
    raise ValueError(f"Puzzles eval supports BC and action-value policies; got: {policy}")

class PuzzlesEvaluator:
  """Runs puzzle accuracy for a local predictor+params."""

  def __init__(
      self,
      predictor: constants.Predictor,
      params: hk.Params,
      config: PuzzlesEvalConfig,
  ) -> None:
    self._predictor = predictor
    self._config = config

    # Resolve puzzles.csv
    if config.puzzles_path is None:
      # repo_root/data/puzzles.csv
      src_dir = Path(__file__).resolve().parent
      self._puzzles_path = src_dir.parent / "data" / "puzzles.csv"
    else:
      self._puzzles_path = Path(config.puzzles_path)

    if not self._puzzles_path.exists():
      raise FileNotFoundError(f"Puzzles CSV not found at {self._puzzles_path}")

    # Build a padded/batched predict_fn (handles arbitrary batch sizes)
    self._predict_fn = neural_engines.wrap_predict_fn(
        predictor=self._predictor,
        params=params,
        batch_size=self._config.batch_size,
    )
    self._engine = _engine_from_policy(
    self._config.policy,
    self._predict_fn,
    self._config.num_return_buckets,
    )

  def step(self) -> Mapping[str, float]:
    """Return dict of metrics for logging (accuracy, counts, avg rating)."""
    n = self._config.num_puzzles
    df = pd.read_csv(self._puzzles_path, nrows=n)

    solved = 0
    ratings_solved = []

    for _, row in df.iterrows():
      pgn = row["PGN"]
      game = chess.pgn.read_game(io.StringIO(pgn))
      if game is None:
        continue
      board = game.end().board()
      moves = row["Moves"].split(" ")
      ok = _evaluate_puzzle_from_board(board=board, moves=moves, engine=self._engine)
      if ok:
        solved += 1
        ratings_solved.append(int(row.get("Rating", 0)))

    total = len(df)
    acc = solved / max(total, 1)
    avg_rating = float(np.mean(ratings_solved)) if ratings_solved else 0.0
    return {
        "puzzles_total": float(total),
        "puzzles_solved": float(solved),
        "puzzles_accuracy": float(acc),
        "puzzles_avg_rating_solved": float(avg_rating),
    }
