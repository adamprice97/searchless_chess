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
from searchless_chess.src import transformer
from searchless_chess.src.engines import neural_engines


@dataclass
class PuzzlesEvalConfig:
  puzzles_path: str | None = None       # default resolves to repo_root/data/puzzles.csv
  num_puzzles: int = 512                # how many to evaluate each time
  batch_size: int = 64                  # for the predict_fn wrapper
  policy: str = "behavioral_cloning_param"  # used to choose engine


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


def _engine_from_policy(policy: str, predict_fn):
  if policy == "behavioral_cloning_param":
    return neural_engines.ParamBCEngine(predict_fn=predict_fn)
  elif policy == "behavioral_cloning":
    return neural_engines.BCEngine(predict_fn=predict_fn)
  else:
    raise ValueError(f"Puzzles eval only supports BC policies; got: {policy}")


class PuzzlesEvaluator:
  """Runs puzzle accuracy for a local predictor+params."""

  def __init__(
      self,
      predictor: constants.Predictor,
      params: hk.Params,
      config: PuzzlesEvalConfig,
      predictor_config: transformer.TransformerConfig
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
    if self._config.policy == "behavioral_cloning_param":
      decoder = transformer.build_param_action_decoder(predictor_config)
      decode_apply = decoder.apply

      self._engine = neural_engines.ParamBCEngine(
        predict_fn=self._predict_fn,   # still available if you want the old ranking mode later
        params=params,                 # EMA params you unreplicate for eval
        decode_apply=decode_apply,
        temperature=None,              # or >0.0 if you want sampling
        greedy=True,                   # False to sample stochastically
      )
    else:
      self._predict_fn = neural_engines.wrap_predict_fn(
       predictor=self._predictor,
       params=params,
       batch_size=self._config.batch_size,
      )
      self._engine = _engine_from_policy(self._config.policy, self._predict_fn)


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
