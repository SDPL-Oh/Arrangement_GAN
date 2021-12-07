from graph_nets import utils_tf
from House_gan_sdpl.utils import (Generator, Discriminator)

import os
import numpy as np
import sonnet as snt
import pandas as pd
import collections
import tensorflow as tf
import matplotlib.pylab as plt

# TODO:
#   데이터가 1000으로 설정되어있을 때, 전체 데이터가 읽어지는지 확인 할 것
class load_tfrecord:
    def readTfrecord(self, example):
        tfrecord_format = (
            {
            'filename':  tf.io.FixedLenFeature((), tf.string),
            'width': tf.io.FixedLenFeature((), tf.float32),
            'height': tf.io.FixedLenFeature((), tf.float32),
            'object/bbox/x': tf.io.VarLenFeature(tf.float32),
            'object/bbox/y': tf.io.VarLenFeature(tf.float32),
            'object/bbox/w': tf.io.VarLenFeature(tf.float32),
            'object/bbox/h': tf.io.VarLenFeature(tf.float32),
            'object/class/text': tf.io.VarLenFeature(tf.string),
            'object/class/label': tf.io.VarLenFeature(tf.int64),
            }
        )
        example = tf.io.parse_single_example(example, tfrecord_format)
        inputs, outputs = self.setData(example)
        return example['height'], example['width'], example['filename'], inputs, outputs

    def sparseFunction(self, *args):
        output_args = []
        for arg in args:
            tmp = tf.sparse.reshape(arg, [-1, 1])
            tmp = tf.sparse.to_dense(tmp)
            output_args.append(tmp)
        return tuple(output_args)

    def setData(self, example):
        x, y, w, h = self.sparseFunction(
            example['object/bbox/x'],
            example['object/bbox/y'],
            example['object/bbox/w'],
            example['object/bbox/h'],
        )
        c = tf.sparse.reshape(example['object/class/label'], [-1])
        c = tf.one_hot(tf.sparse.to_dense(c), depth=10, axis=-1)
        input_data = tf.concat([w, h, c], 1)
        output_data = tf.concat([x, y, w, h, c], 1)
        return input_data, output_data

    def loadData(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.readTfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def getDataset(self, filenames, batch_size, is_training=True):
        dataset = self.loadData(filenames)
        if is_training:
            dataset = dataset.shuffle(1000)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).repeat(3)
        return dataset

    # def randomInput(self, node, num_class, height_range, width_range, batch_size):
    #     batch_node, nodes = [], []
    #     for batch in range(batch_size):
    #         for idx in range(node):
    #             height = np.random.randint(*height_range, size=1)
    #             width = np.random.randint(*width_range, size=1)
    #             classes_vector = tf.one_hot(indices=np.random.randint(num_class, size=1), depth=num_class, axis=-1)
    #             nodes.append([height, width, classes_vector])
    #         batch_node.append(nodes)
    #     return batch_node


class HouseGan:
    def __init__(self, hparams):
        self.input_size = hparams['input_size']
        self.output_size = hparams['output_size']
        self.num_class = hparams['num_class']
        self.latent = hparams['latent']
        self.batch = hparams['batch']
        self.img_size = hparams['img_size']
        self.num_variations = hparams['num_variations']
        self.epochs = hparams['epochs']
        self.generator_lr = hparams['generator_lr']
        self.discriminator_lr = hparams['discriminator_lr']
        self.num_process = hparams['num_process']
        self.decay_steps = hparams['decay_steps']
        self.decay_rate = hparams['decay_rate']
        self.model_path = hparams['model_path']
        self.plt_path = hparams['plt_path']
        self.log_path = hparams['log_path']
        self.train_data = hparams['train_data']
        self.test_data = hparams['test_data']

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.generator_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)
        self.generator = Generator(node_output_size=2)
        self.discriminator = Discriminator(node_output_size=1)
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=self.discriminator_lr)

    def getCombination(self, n, r):
        def factorial(k):
            fact = 1
            for i in range(1, k + 1):
                fact *= i
            return fact
        return np.int64(factorial(n) / (factorial(r) * factorial(n - r)))

    def baseGraphsNp(self, node, purpose):
        if purpose == 'inputs':
            nodes = np.zeros([node, self.input_size + self.num_class + self.latent], np.float32)
            nodes[:, 2+self.num_class:] = np.random.normal(size=(node, self.latent))
        else:
            nodes = np.zeros([node, self.input_size + self.output_size + self.num_class], np.float32)
        edges = np.zeros([np.multiply(self.getCombination(node, 2), 2), 1], np.float32)
        senders, receivers = [], []
        nodes_list = np.linspace(0, node-1, node).astype(int)
        for send in nodes_list:
            for receive in np.delete(nodes_list, send):
                senders.append(send)
                receivers.append(receive)
        return {
            "globals": [0.],
            "nodes": nodes,
            "edges": edges,
            "receivers": receivers,
            "senders": senders
        }

    def baseGraphsTf(self, node, purpose):
        if purpose == 'inputs':
            nodes = tf.zeros([node, 2 + self.num_class + self.latent], tf.float32)
            nodes[:, 2 + self.num_class:] = tf.random.normal([node, self.latent])
        elif purpose == 'target':
            nodes = tf.zeros([node, 4 + self.num_class])
        else:
            nodes = tf.zeros([node, 4 + self.num_class])
        edges = tf.zeros([tf.multiply(self.getCombination(node, 2), 2), 1], tf.float32)
        senders, receivers = [], []
        nodes_list = np.linspace(0, node-1, node).astype(int)
        for send in nodes_list:
            for receive in np.delete(nodes_list, send):
                senders.append(send)
                receivers.append(receive)
        return {
            "globals": [0.],
            "nodes": nodes,
            "edges": edges,
            "receivers": receivers,
            "senders": senders
        }

    def graphTuple(self, nodes, purpose):
        batches_graph = []
        for node in nodes:
            init = self.baseGraphsNp(len(node), purpose)
            if not purpose == 'outputs':
                init['nodes'][:, :node.shape[1]] = node
            batches_graph.append(init)
        input_tuple = utils_tf.data_dicts_to_graphs_tuple(batches_graph)
        return input_tuple

    def generateGraph(self, input_op, output_ops):
        generate_ops = [
            tf.concat([output_op.nodes, input_op.nodes[:, :12]], axis=-1) for output_op in output_ops]
        return tf.stack(generate_ops)

    def oneSingleLoss(self, target_ops, output_ops):
        loss_ops = [self.cross_entropy(tf.ones_like(target_op.nodes), output_op.nodes)
                    for target_op, output_op in zip(target_ops, output_ops)]
        return tf.stack(loss_ops)

    def zeroSingleLoss(self, target_ops, output_ops):
        loss_ops = [self.cross_entropy(tf.zeros_like(target_op.nodes), output_op.nodes)
                    for target_op, output_op in zip(target_ops, output_ops)]
        return tf.stack(loss_ops)

    def averageLoss(self, lbl, prd, one_zero='one'):
        if one_zero == 'zero':
            per_example_loss = self.zeroSingleLoss(lbl, prd)
        else:
            per_example_loss = self.oneSingleLoss(lbl, prd)
        return tf.math.reduce_sum(per_example_loss) / self.num_process

    def initSpec(self):
        init = utils_tf.data_dicts_to_graphs_tuple([self.baseGraphsNp])
        return utils_tf.specs_from_graphs_tuple(init)

    def generatorLoss(self, fake_output):
        return self.averageLoss(fake_output, fake_output, 'one')

    def discriminatorLoss(self, real_output, fake_output):
        real_loss = self.averageLoss(real_output, real_output, 'one')
        fake_loss = self.averageLoss(fake_output, fake_output, 'zero')
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_output = self.generator(x, self.num_process)
            generated_graph = self.generateGraph(x, generated_output)

            output_tuple = self.graphTuple(generated_graph, 'outputs')
            output_tuple_tf = output_tuple.replace(nodes=generated_graph[0])

            real_output = self.discriminator(y, 1)
            fake_output = self.discriminator(output_tuple_tf, 1)

            gen_loss = self.generatorLoss(fake_output)
            disc_loss = self.discriminatorLoss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_opt.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    @tf.function
    def test_step(self, x, y):
        generated_graph = self.generator(x, self.num_process)

    # @tf.function(input_signature=[init_spec()])
    # def predict_step(self, x):
    #     return self.generator(x, self.num_process)
    #
    # @tf.function
    # def valid_step(self, x, y):
    #     return average_loss(y, x)

    # def plot_step(self, x, y, filename, path, x_offset):
    #     x = pd.DataFrame(np.array(x[-1].nodes))
    #     y = pd.DataFrame(np.array(y.nodes))
    #     plt.figure()
    #     plt.axis([0, 1, -0.5, 0.5])
    #     plt.plot(x_offset[0], x[0], linewidth=1, c='r', label='Predict')
    #     plt.plot(x_offset[0], y[0], linewidth=1, c='b', label='Original')
    #     plt.legend()
    #     plt.savefig(plt_path + '{}/{}.png'.format(path, os.path.splitext(np.array(filename)[0].decode())[0]))
    #     plt.close()
    #     x.to_csv(plt_path + '{}/{}.csv'.format(path, os.path.splitext(np.array(filename)[0].decode())[0]))

    def training(self):
        next_batch = load_tfrecord()
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), module=[self.generator, self.discriminator])
        manager = tf.train.CheckpointManager(checkpoint, self.model_path, max_to_keep=3)
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        train_dataset = next_batch.getDataset(self.train_data, self.batch, True)
        # test_dataset = next_batch.get_dataset(self.test_data, 1)
        # space_height, space_width, filename, inputs, outputs = next(iter(train_dataset))

        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % epoch)
            for step, (space_height, space_width, filename, inputs, outputs) in enumerate(train_dataset):
                step_pre_batch = len(filename)
                ############# training step ##############
                checkpoint.step.assign_add(1)

                input_tuple = self.graphTuple(inputs, 'inputs')
                target_tuple = self.graphTuple(outputs, 'target')
                gen_loss, disc_loss = self.train_step(input_tuple, target_tuple)

                if step % 200 == 0:
                    print("Training loss (for %d batch) at step %d: %.8f, %.8f"
                          % (int(step_pre_batch), step, float(gen_loss), float(disc_loss)),
                          "samples: {}".format(filename[-1]),
                          "lr_rate: {:0.6f}".format(self.generator_opt._decayed_lr(tf.float32).numpy()))

        #         if int(step) % 1000 == 0:
        #             ############# save validatoin plot image #############
        #             edges_arrange_val, val_xy_val, filename_val, x_offset = next(iter(test_dataset))
        #             val_input_tuple, val_target_tuple = self.graphTuple(edges_arrange_val,
        #                                                                val_xy_val, 1)
        #             pre = self.predict_step(val_input_tuple)
        #             # plot_step(pre, val_target_tuple, filename_val, 'test', x_offset)
        #             ############# save checkpoint ##############
        #             save_path = manager.save()
        #             print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
        #
        #         if int(step) % 10000 == 0:
        #             ############# save model ##############
        #             to_save = snt.Module()
        #             to_save.inference = self.predict_step
        #             to_save.all_variables = list(self.generator.variables)
        #             tf.saved_model.save(to_save, self.model_path)
        #             print("Saved module for step {}".format(int(step)))


    # def validation(self):
    #     ############# load model ##############
    #     val_batch_size = 1
    #     loaded = tf.saved_model.load(model_path)
    #     ############# dataset ##############
    #     test_dataset = next_batch.get_dataset('dat/val_test.record', val_batch_size, is_training=False)
    #     val_loss = 0
    #     for step_val, (edges_arrange_val, val_xy_val, filename_val, x_offset) in enumerate(test_dataset):
    #         step_pre_batch = len(filename_val)
    #         ############# validation step ##############
    #         val_input_tuple, val_target_tuple = graph_to_tuple(edges_arrange_val, val_xy_val, step_pre_batch)
    #
    #         pre = loaded.inference(val_input_tuple)
    #         plot_step(pre, val_target_tuple, filename_val, 'test', x_offset)
    #         val_loss_per = valid_step(pre, val_target_tuple)
    #         val_loss += val_loss_per
    #         print(val_loss_per.values)
    #         ############## plot offset ##############
    #     print("validation loss: %.8f" % (val_loss / (step_val + 1)))