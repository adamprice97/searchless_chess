# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""An example training script."""

from collections.abc import Sequence

from absl import app
from absl import flags

from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import metrics_evaluator
from searchless_chess.src import tokenizer
from searchless_chess.src import training
from searchless_chess.src import transformer
from searchless_chess.src import utils


_POLICY = flags.DEFINE_enum(
    'policy',
    'action_value',
    config_lib.POLICY_TYPES,
    'The policy used to play moves with the model.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  policy: config_lib.PolicyType = _POLICY.value  # pytype: disable=annotation-type-mismatch
  num_return_buckets = 128

  max_sequence_length = tokenizer.SEQUENCE_LENGTH + 2

  match policy:
    case 'action_value' | 'state_value':
      output_size = num_return_buckets
    case 'behavioral_cloning':
      output_size = utils.NUM_ACTIONS
    case 'behavioral_cloning_param':
      output_size = 64  # unified head vocab for (from/to/promo)
      max_sequence_length += 1

  # === BEHAVIORAL CLONING (BC) — MAIN SETUP ===
  # Model: 16 layers, 8 heads, d_model=1024, learned pos encodings, no causal mask.
  # Data: 10M-game train set. Optimizer: Adam, lr=1e-4, batch=4096, 10M steps.

  predictor_config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,           # keep as-is per your API
      output_size=output_size,          
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=max_sequence_length,     # BC context length
      num_heads=4,
      num_layers=4,
      embedding_dim=256,                     # d_model
      apply_post_ln=True,                     # post-norm + SwiGLU in paper
      apply_qk_layernorm=False,
      use_causal_mask=False,                  # no causal mask
  )

  train_config = config_lib.TrainConfig(
      learning_rate=4e-4,
      data=config_lib.DataConfig(
          batch_size=512,                    # paper main setup
          shuffle=True,
          worker_count=0,
          num_return_buckets=0,               # BC has no value bins
          policy=policy,
          split='train',
      ),
      log_frequency=100,                      # practical default; adjust if noisy
      num_steps=5_000_000,                   # ~2.67 epochs on 10M games
      ckpt_frequency=25_000,                  # sensible cadence
      save_frequency=25_000,
  )

  eval_config = config_lib.EvalConfig(
      data=config_lib.DataConfig(
          batch_size=512,                    # eval throughput
          shuffle=False,
          worker_count=0,
          num_return_buckets=0,               # BC
          policy=None,                        # pytype: disable=wrong-arg-types
          split='test',
      ),
      use_ema_params=True,
      policy=policy,
      batch_size=512,                        # eval micro-batch if your loop uses it
      num_return_buckets=0,
      num_eval_data=10_000,                   # ≈ size of their test set (states)
  )


  params = training.train(
      train_config=train_config,
      predictor_config=predictor_config,
      build_data_loader=data_loader.build_data_loader,
  )

  if policy == 'behavioral_cloning_param':
    predictor = transformer.build_param_action_predictor(predictor_config)
  else:
    predictor = transformer.build_transformer_predictor(predictor_config)

  predictor = transformer.build_transformer_predictor(predictor_config)
  evaluator = metrics_evaluator.build_evaluator(predictor, eval_config)
  print(evaluator.step(params=params, step=train_config.num_steps))


if __name__ == '__main__':
  app.run(main)
