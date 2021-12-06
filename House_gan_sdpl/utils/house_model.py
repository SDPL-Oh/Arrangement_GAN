from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt
import tensorflow as tf


def make_mlp_model():
  return snt.Sequential([
    # snt.nets.MLP([16] * 2, activation=tf.keras.layers.LeakyReLU(), activate_final=True),
    # snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    snt.Linear(8),
    tf.keras.layers.LeakyReLU(),
    snt.Linear(8),
    tf.keras.layers.LeakyReLU(),
    snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

def make_linear_model():
  return snt.Sequential([
    # snt.Linear(1024),
    # tf.keras.layers.LeakyReLU(),
    # snt.Reshape((8, 8, 16))
    snt.Linear(64),
    tf.keras.layers.LeakyReLU(),
    snt.Linear(32),
    tf.keras.layers.LeakyReLU(),
    snt.Linear(8),
    tf.keras.layers.LeakyReLU(),
    snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

def make_conv_model():
  return snt.Sequential([
    snt.Conv2D(
      output_channels=16,
      kernel_shape=(3, 3),
      padding='SAME',
      stride=1),
    tf.keras.layers.LeakyReLU(),
    snt.Conv2D(
      output_channels=16,
      kernel_shape=(3, 3),
      padding='SAME',
      stride=1),
    tf.keras.layers.LeakyReLU(),
    snt.Conv2D(
      output_channels=16,
      kernel_shape=(3, 3),
      padding='SAME',
      stride=1),
    tf.keras.layers.LeakyReLU(),
    snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])


class GeneratorEncoder(snt.Module):
  def __init__(self, name="GeneratorEncoder"):
    super(GeneratorEncoder, self).__init__(name=name)
    self._network = modules.GraphIndependent(
      edge_model_fn=make_mlp_model,
      node_model_fn=make_linear_model,
      global_model_fn=make_mlp_model
    )

  def __call__(self, inputs):
    return self._network(inputs)


class GeneratorDecoder(snt.Module):
  def __init__(self, name="GeneratorDecoder"):
    super(GeneratorDecoder, self).__init__(name=name)
    self._network = modules.GraphIndependent(
      edge_model_fn=make_mlp_model,
      node_model_fn=make_mlp_model,
      global_model_fn=make_mlp_model)

  def __call__(self, inputs):
    return self._network(inputs)


class GeneratorCore(snt.Module):
  def __init__(self, name="GeneratorCore"):
    super(GeneratorCore, self).__init__(name=name)
    self._network = modules.GraphNetwork(make_mlp_model,
                                         make_mlp_model,
                                         make_mlp_model)

  def __call__(self, inputs):
    return self._network(inputs)


class Generator(snt.Module):
  def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="Generator"):
    super(Generator, self).__init__(name=name)
    self._encoder = GeneratorEncoder()
    self._core = GeneratorCore()
    self._second_core = GeneratorCore()
    self._decoder = GeneratorDecoder()
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    self._output_transform = modules.GraphIndependent(
        edge_fn, node_fn, global_fn)

  def __call__(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
      decoded_op = self._decoder(latent)
      output_ops.append(self._output_transform(decoded_op))
    return output_ops


class DiscriminatorEncoder(snt.Module):
  def __init__(self, name="GeneratorEncoder"):
    super(DiscriminatorEncoder, self).__init__(name=name)
    self._network = modules.GraphIndependent(
      edge_model_fn=make_mlp_model,
      node_model_fn=make_mlp_model,
      global_model_fn=make_mlp_model
    )

  def __call__(self, inputs):
    return self._network(inputs)


class DiscriminatorDecoder(snt.Module):
  def __init__(self, name="GeneratorDecoder"):
    super(DiscriminatorDecoder, self).__init__(name=name)
    self._network = modules.GraphIndependent(
      edge_model_fn=make_mlp_model,
      node_model_fn=make_mlp_model,
      global_model_fn=make_mlp_model)

  def __call__(self, inputs):
    return self._network(inputs)


class DiscriminatorCore(snt.Module):
  def __init__(self, name="GeneratorCore"):
    super(DiscriminatorCore, self).__init__(name=name)
    self._network = modules.GraphNetwork(make_mlp_model,
                                         make_mlp_model,
                                         make_mlp_model)

  def __call__(self, inputs):
    return self._network(inputs)


class Discriminator(snt.Module):
  def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="Generator"):
    super(Discriminator, self).__init__(name=name)
    self._encoder = GeneratorEncoder()
    self._core = GeneratorCore()
    self._second_core = GeneratorCore()
    self._decoder = GeneratorDecoder()
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    self._output_transform = modules.GraphIndependent(
        edge_fn, node_fn, global_fn)

  def __call__(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
      decoded_op = self._decoder(latent)
      output_ops.append(self._output_transform(decoded_op))
    return output_ops