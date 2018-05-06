from utils import create_dir, one_hot, pickle_load, pickle_save
import tensorflow as tf
import numpy as np
import logging
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
from os.path import join
import re
import configparser

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DCNN1D():
    def __init__(self, istest=False, path_model=''):
        # Create or load configuration file
        if istest is False:
            logger.info("Creating configuration start...")
            config = configparser.ConfigParser()
            config.read('configuration.ini')
            self.seq_length = int(config['DEFAULT']['seq_length'])
            self.n_channels = int(config['DEFAULT']['n_channels'])
            self.n_classes = int(config['DEFAULT']['n_classes'])
            self.SAVE_DIR = config['DEFAULT']['SAVE_DIR']

            self.num_epochs = int(config['TRAIN']['num_epochs'])
            self.batch_size = int(config['TRAIN']['batch_size'])
            self.n_batches = int(config['TRAIN']['n_batches'])
            self.log_every = int(config['TRAIN']['log_every'])
            self.val_every = int(config['TRAIN']['val_every'])

            self.filters = [int(chunk) for chunk in config['CNN']['filters'].split(',')]
            self.kernels_size = [int(chunk) for chunk in config['CNN']['kernels_size'].split(',')]

            self.grad_clip = int(config['HIPERPARAMETERS']['grad_clip'])
            self.decay_rate = float(config['HIPERPARAMETERS']['decay_rate'])
            self.lr = float(config['HIPERPARAMETERS']['lr'])
            self.keep_prob = float(config['HIPERPARAMETERS']['keep_prob'])

            self.path_classes = config['DATASET']['path_classes']
            self.name_acq = config['DATASET']['name_acq']
            self.sensor_number = int(config['DATASET']['sensor_number'])

            create_dir(self.SAVE_DIR)
            with open(self.SAVE_DIR+'configuration.ini', 'w') as configfile:
                config.write(configfile)
            logger.info("Creating configuration done...")

            logger.info("Building dataset start...")
            self.signals_train, self.labels_train,\
            self.signals_val, self.labels_val,\
            self.signals_test, self.labels_test = self.preprocessing()
            pickle_save((self.signals_train, self.labels_train), self.SAVE_DIR + 'train_set.pkl')
            pickle_save((self.signals_val, self.labels_val), self.SAVE_DIR + 'val_set.pkl')
            pickle_save((self.signals_test, self.labels_test), self.SAVE_DIR + 'test_set.pkl')
            logger.info("Building dataset done.")
        else:
            logger.info("Loading configuration and datasets start...")
            config = configparser.ConfigParser()
            config.read(path_model+'configuration.ini')
            self.seq_length = int(config['DEFAULT']['seq_length'])
            self.n_channels = int(config['DEFAULT']['n_channels'])
            self.n_classes = int(config['DEFAULT']['n_classes'])
            self.SAVE_DIR = config['DEFAULT']['SAVE_DIR']

            self.num_epochs = int(config['TRAIN']['num_epochs'])
            self.batch_size = int(config['TRAIN']['batch_size'])
            self.n_batches = int(config['TRAIN']['n_batches'])
            self.log_every = int(config['TRAIN']['log_every'])
            self.val_every = int(config['TRAIN']['val_every'])

            self.filters = [ int(chunk) for chunk in config['CNN']['filters'].split(',') ]
            self.kernels_size = [ int(chunk) for chunk in config['CNN']['kernels_size'].split(',') ]

            self.grad_clip = int(config['HIPERPARAMETERS']['grad_clip'])
            self.decay_rate = float(config['HIPERPARAMETERS']['decay_rate'])
            self.lr = float(config['HIPERPARAMETERS']['lr'])
            self.keep_prob = float(config['HIPERPARAMETERS']['keep_prob'])

            self.path_classes = config['DATASET']['path_classes']
            self.name_acq = config['DATASET']['name_acq']
            self.sensor_number = int(config['DATASET']['sensor_number'])
            self.batch_size = 1
            self.signals_train, self.labels_train = pickle_load(self.SAVE_DIR+'train_set.pkl')
            self.signals_val, self.labels_val = pickle_load(self.SAVE_DIR + 'val_set.pkl')
            self.signals_test, self.labels_test = pickle_load(self.SAVE_DIR + 'test_set.pkl')
            logger.info("Loading configuration and datasets done.")


        logger.info("Building model start...")
        # zona sub-funciones
        # zona if test

        self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_length, self.n_channels], name='inputs')
        self.labels_ = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')

        cnn_layer = self.inputs_
        for i, filter in enumerate(self.filters):
            cnn_layer = tf.layers.conv1d(inputs=cnn_layer, filters=filter, kernel_size=self.kernels_size[i], strides=1,
                                         padding='same', activation=tf.nn.relu)
            cnn_layer = tf.layers.max_pooling1d(inputs=cnn_layer, pool_size=2, strides=2, padding='same')

        '''
        conv1 = tf.layers.conv1d(inputs=self.inputs_, filters=18, kernel_size=100, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

        # (batch, 64, 18) --> (batch, 32, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=100, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

        # (batch, 32, 36) --> (batch, 16, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=100, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

        # (batch, 16, 72) --> (batch, 8, 144)
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=100, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
        '''
        flat = tf.reshape(cnn_layer, (-1, int(self.filters[-1] * self.seq_length / (2 ** len(self.filters)))))
        flat = tf.nn.dropout(flat, keep_prob=self.keep_prob)

        # Predictions
        logits = tf.layers.dense(flat, self.n_classes)

        # Cost function and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels_))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        # Accuracy
        self.pred_class = tf.argmax(logits, 1)
        correct_pred = tf.equal(self.pred_class, tf.argmax(self.labels_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        self.lr = tf.Variable(self.lr, trainable=False)
        logger.info("Building model done.")

        self.sess = tf.Session()

    def next_batch(self, stage='Train'):
        if stage == 'Train':
            signals = self.signals_train
            labels = self.labels_train
        elif stage == 'Val':
            signals = self.signals_val
            labels = self.labels_val
        elif stage == 'Test':
            signals = self.signals_test
            labels = self.labels_test

        n_signals, total_len = signals.shape
        random_idx = np.random.randint(low=0, high=n_signals, size=self.batch_size)
        random_pos = np.random.randint(low=0, high=total_len - self.seq_length, size=self.batch_size)
        x = np.array([signals[random_idx[i]][random_pos[i]:random_pos[i] + self.seq_length]
                      for i in range(self.batch_size)])[:, :, np.newaxis]
        y = labels[random_idx]
        return x, y

    def all_subsignals(self, stage='Train'):
        if stage == 'Train':
            signals = self.signals_train
            labels = self.labels_train
        elif stage == 'Val':
            signals = self.signals_val
            labels = self.labels_val
        elif stage == 'Test':
            signals = self.signals_test
            labels = self.labels_test

        n_signals, total_len = signals.shape
        i = 0
        subsignals = []
        sublabels = []
        while i + self.seq_length < total_len:
            subsignals.append(signals[:,i:i + self.seq_length])
            sublabels.append(labels)
            i += self.seq_length
        x = np.array(subsignals)
        y = np.array(sublabels)
        return x, y

    def initialize(self):
        logger.info("Initialization of parameters")
        self.sess.run(tf.global_variables_initializer())

    def restore(self):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)
        print("Load the model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def preprocessing(self):

        # Reading of signal files
        signals = []
        labels = []
        for dir in os.listdir(self.path_classes):
            folder = join(self.path_classes, dir)
            for file in os.listdir(folder):
                data = sio.loadmat(join(folder, file))
                signals.append(data['data'][self.name_acq][0][0][self.sensor_number])
                label = re.split('R|F|L|P|.mat', file)[-2]
                labels.append(int(label))

        signals = np.array(signals)
        labels = np.array(labels).squeeze()

        # Split on train, val and test datasets
        X_tr, X_vld, lab_tr, lab_vld = train_test_split(signals, labels, test_size=0.3, stratify=labels)
        X_vld, X_test, lab_vld, lab_test = train_test_split(X_vld, lab_vld, test_size=0.5, stratify=lab_vld)

        # One hot representation of labels
        y_tr = one_hot(lab_tr, n_class=self.n_classes)
        y_vld = one_hot(lab_vld, n_class=self.n_classes)
        y_test = one_hot(lab_test, n_class=self.n_classes)

        # Standarization according to mean and std of train dataset
        mean_val = X_tr.mean()
        std_val = X_tr.std()
        X_tr = (X_tr - mean_val) / std_val
        X_vld = (X_vld - mean_val) / std_val
        X_test = (X_test - mean_val) / std_val

        return X_tr, y_tr, X_vld, y_vld, X_test, y_test

    def train(self):
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)
        saver = tf.train.Saver(tf.global_variables())

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load the model from %s" % ckpt.model_checkpoint_path)

        iteration = 0
        for epoch in range(self.num_epochs):
            # Learning rate decay
            self.sess.run(tf.assign(self.lr, self.lr * (self.decay_rate ** epoch)))

            for batch in range(self.n_batches):
                x, y = self.next_batch()
                feed_dict = {model.inputs_: x, model.labels_: y}
                train_loss, _, acc = self.sess.run([self.cost, self.optimizer, self.accuracy], feed_dict=feed_dict)

                iteration += 1

                if iteration % self.log_every == 0 and iteration > 0:
                    print("{}/{}(epoch {}), train_loss = {:.6f}, acc = {:.3f}".format(iteration,
                                                                                      self.num_epochs * self.n_batches,
                                                                                      epoch + 1, train_loss, acc))

                if iteration % self.val_every == 0 and iteration > 0:
                    x, y = self.next_batch(stage='Val')
                    feed_dict = {model.inputs_: x, model.labels_: y}
                    val_loss, acc = self.sess.run([self.cost, self.accuracy], feed_dict=feed_dict)
                    print("{}/{}(epoch {}), val_loss = {:.6f}, acc = {:.3f}".format(iteration,
                                                                                    self.num_epochs * self.n_batches,
                                                                                    epoch + 1, val_loss, acc))
                    checkpoint_path = os.path.join(self.SAVE_DIR, 'model.ckpt')
                    saver.save(self.sess, checkpoint_path, global_step=iteration)
                    logger.info("model saved to {}".format(checkpoint_path))

        x, y = self.next_batch(stage='Test')
        feed_dict = {model.inputs_: x, model.labels_: y}
        test_loss, acc = self.sess.run([self.cost, self.accuracy], feed_dict=feed_dict)
        print("test_loss = {:.6f}, acc = {:.3f}".format(test_loss, acc))
        checkpoint_path = os.path.join(self.SAVE_DIR, 'model.ckpt')
        saver.save(self.sess, checkpoint_path, global_step=iteration)
        logger.info("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    model = DCNN1D()
    model.initialize()
    model.train()
