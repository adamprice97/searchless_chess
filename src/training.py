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

"""Implements the training loop that performs parameter updates."""

import copy
import functools
import os

from absl import logging
import haiku as hk
import jax
from jax.experimental import mesh_utils
import jax.random as jrandom
import numpy as np
import optax

from searchless_chess.src import config as config_lib
from searchless_chess.src import constants
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import puzzles_evaluator
from searchless_chess.src import metrics_evaluator

def _shape_dict(p):
    """Return { 'module/sub/param': shape } for a Haiku params pytree."""
    def _flatten(d, prefix=()):
        if isinstance(d, dict):
            for k, v in d.items():
                yield from _flatten(v, prefix + (k,))
        else:
            yield "/".join(prefix), getattr(d, "shape", ())
    # hk.data_structures.to_mutable_dict works across HK versions
    flat = hk.data_structures.to_mutable_dict(p)
    return dict(_flatten(flat))


def train(
    train_config: config_lib.TrainConfig,
    predictor_config: transformer.TransformerConfig,
    build_data_loader: constants.DataLoaderBuilder,
) -> hk.Params:
  """Trains a predictor and returns the trained parameters."""
  logging.info(
      '[Process %d]: Using %d processes with %d local devices each.',
      jax.process_index(),
      jax.process_count(),
      jax.local_device_count(),
  )

  # Build the predictor and the data loader.
  if train_config.data.policy == 'behavioral_cloning_param':
    predictor = transformer.build_param_action_predictor(predictor_config)
    dummy_len = 3
  else:
    predictor = transformer.build_transformer_predictor(predictor_config)
    dummy_len = 1

  puzzles_eval_cfg = puzzles_evaluator.PuzzlesEvalConfig(
      puzzles_path=getattr(train_config, "puzzles_path", None),
      num_puzzles=getattr(train_config, "puzzles_num", 2048),
      batch_size=getattr(train_config, "puzzles_batch_size", 64),
      policy=train_config.data.policy,  # 'behavioral_cloning_param' etc.
    )

  evaluator = metrics_evaluator.build_evaluator(
        predictor=predictor,
        config=train_config.eval,   # config_lib.EvalConfig
    )

  # For multi-host topologies, we want every process to train on different data,
  # so we need to modify the seed accordingly.
  train_config.data.seed += jax.process_index()
  data_iter = build_data_loader(config=train_config.data).__iter__()

  # Initialize the predictor parameters.
  logging.info('Initializing the predictor parameters.')
  
  dummy_len = 31
  dummy_targets = (np.arange(dummy_len, dtype=np.uint32)[None, :]
                    % predictor_config.vocab_size)
  params = predictor.initial_params(
        rng=jrandom.PRNGKey(predictor_config.seed),
        targets=dummy_targets,
  )
  params_ema = copy.deepcopy(params)
  # Create the optimizer and initialize its state.
  optimizer = optax.chain(
      optax.clip_by_global_norm(train_config.max_grad_norm),
      optax.adam(train_config.learning_rate),
  )
  opt_state = optimizer.init(params)

  # Create the gradient and update functions.
  loss_fn = training_utils.make_loss_fn(predictor=predictor)
  grad_fn = jax.value_and_grad(loss_fn)
  update_fn = functools.partial(
      training_utils.update_parameters,
      grad_fn=grad_fn,
      optimizer=optimizer,
  )

  # Create the sharding and replicate the parameters and the optimizer state.
  # The sharding is flat, i.e., all devices are used in parallel (disregarding
  # where they are on the grid).
  devices = mesh_utils.create_device_mesh((jax.device_count(),))
  sharding = jax.sharding.PositionalSharding(devices)
  sharding = sharding.reshape((jax.device_count(), 1))
  params = training_utils.replicate(params, sharding)
  params_ema = training_utils.replicate(params_ema, sharding)
  opt_state = training_utils.replicate(opt_state, sharding)

  latest_step = 0

  # Initialize the checkpointer and restore any previous checkpoints.
  if train_config.ckpt_frequency is not None:
    logging.info('Initializing the checkpoint manager.')
    checkpoint_dir = os.path.join(
        os.getcwd(),
        f'checkpoints/local/{train_config.data.policy}',
    )
    checkpoint_manager = training_utils.get_checkpoint_manager(
        ckpt_frequency=train_config.ckpt_frequency,
        max_to_keep=train_config.ckpt_max_to_keep,
        save_frequency=train_config.save_frequency,
        checkpoint_dir=checkpoint_dir,
    )

    if checkpoint_manager.latest_step() is not None:
      latest_step = checkpoint_manager.latest_step()
      logging.info('Restoring checkpoint %d.', latest_step)
      params, params_ema, opt_state, data_iter = (
          training_utils.restore_checkpoint(
              checkpoint_manager=checkpoint_manager,
              step=latest_step,
              params=params,
              params_ema=params_ema,
              opt_state=opt_state,
              data_iter=data_iter,
              sharding=sharding,
          )
      )

  # Main training loop.
  for step in range(latest_step, train_config.num_steps + 1):
    if train_config.ckpt_frequency is not None:
      if step % train_config.ckpt_frequency == 0:
        logging.info('Checkpointing step %i.', step)
        checkpoint_manager.save(
            step=step,
            items=dict(
                params=params,
                params_ema=params_ema,
                opt_state=opt_state,
                data_iter=data_iter,
            ),
        )

    sequences, loss_mask = next(data_iter)
    sequences = jax.lax.with_sharding_constraint(sequences, sharding)
    loss_mask = jax.lax.with_sharding_constraint(loss_mask, sharding)

    params, params_ema, opt_state, loss, grad_norm_unclipped = update_fn(
        params=params,
        params_ema=params_ema,
        opt_state=opt_state,
        sequences=sequences,
        loss_mask=loss_mask,
    )

    if train_config.log_frequency is not None:
      if step % train_config.log_frequency == 0:
        logging.info(
            'step: %d | loss: %f | grad_norm_unclipped: %f',
            step,
            jax.device_get(loss),
            jax.device_get(grad_norm_unclipped),
        )

    if puzzles_eval_cfg is not None and (step % train_config.puzzles_eval_every == 0):
        # Unreplicate EMA params for eval
        host_params_ema = training_utils.unreplicate(params_ema)
        # Rebuild the puzzles evaluator with current EMA params
        pe = puzzles_evaluator.PuzzlesEvaluator(
            predictor=predictor,
            params=host_params_ema,
            config=puzzles_eval_cfg,  # the config object
        )
        pz_metrics = pe.step()
        logging.info("PUZZLES EVAL step=%d %s", step,
                    {k: float(v) for k, v in pz_metrics.items()})

        #just chucking this here for now
        host_params_ema = training_utils.unreplicate(params_ema)
        metrics = evaluator.step(params=host_params_ema, step=step)
        # Log a compact line; evaluator returns a dict of numpy scalars.
        logging.info("EVAL step=%d %s", step, {k: float(v) for k, v in metrics.items()})

  if train_config.ckpt_frequency is not None:
    checkpoint_manager.close()

  return jax.device_get(params)
