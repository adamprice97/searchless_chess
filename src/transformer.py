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

"""Transformer model."""

import dataclasses
import enum
import functools

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from searchless_chess.src import constants


class PositionalEncodings(enum.Enum):
  SINUSOID = enum.auto()
  LEARNED = enum.auto()


@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""

  # The random seed for parameter initialization.
  seed: int = 1
  # The input vocabulary size.
  vocab_size: int
  # The output size (by default equal to the vocabulary size).
  output_size: int | None = None
  # The dimension of the first embedding.
  embedding_dim: int = 64
  # The number of multi-head attention layers.
  num_layers: int = 4
  # The number of heads per layer.
  num_heads: int = 8
  # Whether to use a causal mask or not.
  use_causal_mask: bool = True
  # The parameter initialization scale for the embeddings.
  emb_init_scale: float = 0.02
  # Positional encodings to use.
  pos_encodings: PositionalEncodings = PositionalEncodings.SINUSOID
  # Maximum sequence length, useful for the LEARNED positional encodings.
  max_sequence_length: int | None = None
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4
  # Whether to apply QK normalization trick in attention layer.
  apply_qk_layernorm: bool = False
  # Whether to apply post LN after attention + MLP blocks
  apply_post_ln: bool = True

  def __post_init__(self):
    if self.output_size is None:
      self.output_size = self.vocab_size

class FixedEmbed(hk.Module):
    """Embedding layer that always allocates [vocab_size, embed_dim]."""
    def __init__(self, vocab_size: int, embed_dim: int, emb_init_scale: float, name: str):
        super().__init__(name=name)  # e.g., "state_token_embed"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.emb_init = hk.initializers.TruncatedNormal(stddev=emb_init_scale)

    def __call__(self, ids: jax.Array) -> jax.Array:
        # Canonical leaf name: "embeddings"
        embeddings = hk.get_parameter(
            "embeddings",
            shape=[self.vocab_size, self.embed_dim],
            init=self.emb_init,
        )
        return jnp.take(embeddings, ids, axis=0)


class MultiHeadDotProductAttention(hk.Module):
  """Multi-head dot-product attention (Vaswani et al., 2017)."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      name: str | None = None,
      apply_qk_layernorm: bool = False,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      name: Name of the module.
      apply_qk_layernorm: Applies layernorm to query and key matrices, this
        helps training stability.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self._apply_qk_layernorm = apply_qk_layernorm

  def __call__(
      self,
      inputs_q: jax.Array,
      inputs_kv: jax.Array,
      mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns the output of the multi-head attention."""
    batch_size, sequence_length, embedding_size = inputs_q.shape

    num_hiddens = self._num_hiddens_per_head * self._num_heads
    q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
    k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)

    if self._apply_qk_layernorm:
      q = layer_norm(q)
      k = layer_norm(k)

    v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    # The second (sequence) dimension is undefined since it can differ between
    # queries and keys/values when decoding. Also checking that the inputs have
    # the same batch size as the reshape below does not guarantee a failure if
    # they are different.
    new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
    q = jnp.reshape(q, new_shape)
    k = jnp.reshape(k, new_shape)
    v = jnp.reshape(v, new_shape)

    # Let b=batch_size, t=seq_len, h=num_heads, and d=num_hiddens_per_head.
    attention = jnp.einsum('bthd,bThd->bhtT', q, k)
    attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

    if mask is not None:
      attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

    normalized_attention = jnn.softmax(attention)

    output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
    output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
    return hk.Linear(embedding_size, with_bias=False)(output)


def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> np.ndarray:
  """Creates sinusoidal encodings from the original transformer paper.

  The returned values are, for all i < D/2:
    array[pos, i] = sin(pos / (max_timescale^(2*i / D)))
    array[pos, D/2 + i] = cos(pos / (max_timescale^(2*i / D)))

  Args:
    sequence_length: Sequence length.
    hidden_size: Dimension of the positional encoding vectors, D. Should be
      even.
    max_timescale: Maximum timescale for the frequency.

  Returns:
    An array of shape [L, D] if `add_negative` or `keep_positive_side` is
    `False`, else [2 * L, D].
  """
  freqs = np.arange(0, hidden_size + 1, 2)
  inv_freq = max_timescale ** (-freqs / hidden_size)

  pos_seq = np.arange(start=0, stop=sequence_length)

  sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
  embeddings = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
  )
  return embeddings[:, :hidden_size]

def embed_sequences(
    sequences: jax.Array,
    config: TransformerConfig,
    name: str = "token_embed",
) -> jax.Array:
  # Force-allocate full table under its own module scope (name/embeddings)
  token_embeddings = FixedEmbed(
      vocab_size=config.vocab_size,
      embed_dim=config.embedding_dim,
      emb_init_scale=config.emb_init_scale,
      name=name,              
  )
  embeddings = token_embeddings(sequences)
  embeddings = embeddings * jnp.sqrt(config.embedding_dim)

  # Positional encodings (unchanged)
  _, sequence_length, embedding_size = embeddings.shape
  match config.pos_encodings:
    case PositionalEncodings.SINUSOID:
      pos_encodings = sinusoid_position_encoding(
          sequence_length=sequence_length,
          hidden_size=embedding_size,
      )
    case PositionalEncodings.LEARNED:
      assert sequence_length <= config.max_sequence_length
      positions = jnp.arange(sequence_length)
      pos_encodings = hk.Embed(
          vocab_size=config.max_sequence_length,
          embed_dim=embedding_size,
          name="pos_embed",
      )(positions)
  return embeddings + pos_encodings


def layer_norm(x: jax.Array) -> jax.Array:
  """Helper function for layer norm."""
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def shift_right(sequences: jax.Array) -> jax.Array:
  """Right-shift the one-hot encoded input by padding on the temporal axis."""
  bos_array = jnp.zeros((sequences.shape[0], 1), dtype=jnp.uint8)
  padded_sequences = jnp.concatenate([bos_array, sequences], axis=1)
  return padded_sequences[:, :-1]


def _mlp_block(inputs: jax.Array, config: TransformerConfig) -> jax.Array:
  """Gated MLP block for the Transformer."""
  ffn_dim = config.embedding_dim * config.widening_factor
  split_1 = hk.Linear(ffn_dim, with_bias=False)(inputs)
  split_2 = hk.Linear(ffn_dim, with_bias=False)(inputs)
  gate_output = jnn.silu(split_1) * split_2
  return hk.Linear(config.embedding_dim, with_bias=False)(gate_output)


def _attention_block(inputs: jax.Array, config: TransformerConfig) -> jax.Array:
  """Attention block for the Transformer."""
  batch_size, sequence_length = inputs.shape[:2]
  if config.use_causal_mask:
    causal_mask = np.tril(
        np.ones((batch_size, 1, sequence_length, sequence_length))
    )
  else:
    causal_mask = None
  block = MultiHeadDotProductAttention(
      num_heads=config.num_heads,
      num_hiddens_per_head=config.embedding_dim // config.num_heads,
      apply_qk_layernorm=config.apply_qk_layernorm,
  )
  return block(inputs_q=inputs, inputs_kv=inputs, mask=causal_mask)


def transformer_decoder(
    targets: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """Returns the transformer decoder output, shape [B, T, V].

  Follows the LLaMa architecture:
  https://github.com/facebookresearch/llama/blob/main/llama/model.py
  Main changes to the original Transformer decoder:
  - Using gating in the MLP block, with SwiGLU activation function.
  - Using normalization before the attention and MLP blocks.

  Args:
    targets: The integer target values, shape [B, T].
    config: The config to use for the transformer.
  """
  # Right shift the targets to get the inputs (the first token is now a 0).
  inputs = shift_right(targets)

  # Embeds the inputs and adds positional encodings.
  embeddings = embed_sequences(inputs, config)

  h = embeddings
  for _ in range(config.num_layers):
    attention_input = layer_norm(h)
    attention = _attention_block(attention_input, config)
    h += attention

    mlp_input = layer_norm(h)
    mlp_output = _mlp_block(mlp_input, config)
    h += mlp_output

  if config.apply_post_ln:
    h = layer_norm(h)
  logits = hk.Linear(config.output_size)(h)
  return jnn.log_softmax(logits, axis=-1)


def build_transformer_predictor(
    config: TransformerConfig,
) -> constants.Predictor:
  """Returns a transformer predictor."""
  model = hk.transform(functools.partial(transformer_decoder, config=config))
  return constants.Predictor(initial_params=model.init, predict=model.apply)

def _encode_state_core(targets: jax.Array, config: TransformerConfig) -> jax.Array:
  """Embeds the state tokens (everything before the last 3) into a single vector."""
  # targets shape: [B, T_total]; state length = T_total - 3
  state_len = targets.shape[1] - 3
  state_tokens = targets[:, :state_len]  # [B, Ts]
  # Standard embedding + (non-causal) transformer layers to get a core.
  # We reuse embed_sequences for positional encodings.
  x = embed_sequences(state_tokens, config, name='state_token_embed')  # [B, Ts, D]

  # Use the same block stack but without causal masking.
  non_causal_cfg = dataclasses.replace(config, use_causal_mask=False)
  h = x
  for _ in range(non_causal_cfg.num_layers):
    attention_input = layer_norm(h)
    attention = _attention_block(attention_input, non_causal_cfg)
    h += attention
    mlp_input = layer_norm(h)
    mlp_output = _mlp_block(mlp_input, non_causal_cfg)
    h += mlp_output
  if non_causal_cfg.apply_post_ln:
    h = layer_norm(h)

  # Pool to a single core vector; mean-pool works well here.
  core = jnp.mean(h, axis=1)  # [B, D]
  return core

def param_action_heads(
    targets: jax.Array,  # [B, T] = (state tokens) + [from, to, promo]
    config: TransformerConfig,
) -> jax.Array:
  """Returns log-probs shaped [B, T, V] with V=64; last 3 steps are valid.

  - Step T-3: logits over from-squares (64).
  - Step T-2: logits over to-squares (64), conditioned on sampled/teacher-forced 'from'.
  - Step T-1: logits over promotions (first 5 indices), conditioned on from & to.
  """
  B, T = targets.shape
  V = 64  # unified vocabulary for the heads

  # Compute core from the state.
  core = _encode_state_core(targets, config)  # [B, D]

  # Teacher-forcing: use the ground-truth 'from' and 'to' tokens for conditioning.
  from_gt = targets[:, -3]            # [B], ints in [0,63]
  to_gt = targets[:, -2]              # [B], ints in [0,63]
  from_oh = jax.nn.one_hot(from_gt, 64)  # [B, 64]
  to_oh   = jax.nn.one_hot(to_gt, 64)    # [B, 64]

  # Head 1: p(from)
  h1 = jnp.concatenate([core], axis=-1)
  logits_from = hk.Linear(V)(h1)  # [B, 64]

  # Head 2: p(to | from)
  h2 = jnp.concatenate([core, from_oh], axis=-1)
  h2 = hk.Linear(core.shape[-1])(h2); h2 = jnn.silu(h2)
  logits_to = hk.Linear(V)(h2)  # [B, 64]

  # Head 3: p(promo | from, to) -> first 5 classes used
  h3 = jnp.concatenate([core, from_oh, to_oh], axis=-1)
  h3 = hk.Linear(core.shape[-1])(h3); h3 = jnn.silu(h3)
  logits_promo5 = hk.Linear(5)(h3)       # [B, 5]
  # Pad to 64 so we can return [B, 64]; unused classes won't be targeted.
  pad = jnp.full((B, V - 5), -1e9, dtype=logits_promo5.dtype)
  logits_promo = jnp.concatenate([logits_promo5, pad], axis=-1)  # [B, 64]

  # Now place the three heads into the final [B, T, V] tensor.
  logits = jnp.zeros((B, T, V), dtype=logits_from.dtype)
  logits = logits.at[:, -3, :].set(logits_from)
  logits = logits.at[:, -2, :].set(logits_to)
  logits = logits.at[:, -1, :].set(logits_promo)

  return jnn.log_softmax(logits, axis=-1)


def build_param_action_predictor(
    config: TransformerConfig,
) -> constants.Predictor:
  """Predictor for the parameterised BC head, compatible with the trainer."""
  def forward(targets: jax.Array) -> jax.Array:
    return param_action_heads(targets=targets, config=config)
  model = hk.transform(forward)
  return constants.Predictor(initial_params=model.init, predict=model.apply)
