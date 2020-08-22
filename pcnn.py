import argparse
import math
import os
import pickle

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile

np.random.seed(3)

root_path = '.'
parser = argparse.ArgumentParser(description='Relation Exetraction Parameters Parser')
parser.add_argument('-data_path', default=os.path.join(root_path, 'data'), type=str,
                    help='train/eval dataset path')
parser.add_argument('-vocab_path', default=os.path.join(root_path, 'data/vocab.pkl'), type=str,
                    help='vocabulary path')
parser.add_argument('-label_path', default=os.path.join(root_path, 'data/label.pkl'), type=str,
                    help='label path')
parser.add_argument('-output_path', default=os.path.join(root_path, 'output'), type=str,
                    help='output path for saving model')
parser.add_argument('-use_pretrained_vec', default=True, type=bool, help='use pretrained vector')
parser.add_argument('-use_glove_vec', default=False, type=bool, help='use glove vector')
parser.add_argument('-glove_path', default='/content/drive/My Drive/model/glove/glove.6B.200d.txt',
                    type=str, help='glove vector path')
parser.add_argument('-feature_maps', default=128, type=int, help='number of filters')
parser.add_argument('-word_embedding_size', default=50, type=int, help='word embedding size')
parser.add_argument('-pos_embedding_size', default=10, type=int, help='positional embedding size')
parser.add_argument('-max_len', default=256, type=int, help='max sequence length of inputs')
parser.add_argument('-pos_limit', default=100, type=int,
                    help='max distance betweet entity and other word of a sentence')
parser.add_argument('-batch_size', default=64, type=int, help='batch size of training')
parser.add_argument('-init_lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('-end_lr', default=1e-5, type=float, help='end learning rate')
parser.add_argument('-epochs', default=20, type=int, help='training epochs')
parser.add_argument('-mlt_warmup', default=0.1, type=float,
                    help='percentage steps that not use multi instance learning')
parser.add_argument('-lr_warmup', default=0.1, type=float,
                    help='percentage steps that use warmup learning rate')
parser.add_argument('-dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('-do_train', default=True, type=bool, help='training the model')
parser.add_argument('-do_predict', default=True, type=bool, help='predict the test dataset')

# parser.add_argument('-vocab_size', default=0, type=int, help='number of words in vocab')
# parser.add_argument('-n_classes', default=0, type=int, help='number of classes')

args = parser.parse_args(args=[])


def build_corpus(corpus_file, vocab_path, label_path):
    """建立词语字典、关系字典"""
    vocab, labels = {'<PAD>': 0, '<UNK>': 1}, {'NA': 0}
    with open(corpus_file, encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().split()
            labels.setdefault(line[4], len(labels))
            for w in line[5:]:
                vocab.setdefault(w, len(vocab))

    pickle.dump(vocab, open(vocab_path, 'wb'))
    pickle.dump(labels, open(label_path, 'wb'))


def positional_indices(entity_id, length, limit):
    """位置嵌入, 嵌入矩阵维度: (2 * limit + 2, embedding_size)
    -limit ~ limit -> 1 ~ 2*limit+1
    special situation: <-limit：1, >limit: 2*limit+1, padding: 0
    """
    indices = np.arange(-entity_id + limit + 1, length - entity_id + limit + 1)
    indices[indices < 1] = 1
    indices[indices > 2 * limit + 1] = 2 * limit + 1
    return indices


def encode(sentence, entity1, entity2, pos_limit, vocab):
    """编码输入"""
    # sentence to word indices
    word_ids = [vocab.get(w, vocab['<UNK>']) for w in sentence]
    ent_ids = sentence.index(entity1), sentence.index(entity2)
    # positional indices
    pos1_ids = positional_indices(ent_ids[0], len(word_ids), pos_limit)
    pos2_ids = positional_indices(ent_ids[1], len(word_ids), pos_limit)
    return word_ids, ent_ids, pos1_ids, pos2_ids


def load_nyt_dataset(file, max_len, pos_limit, vocab, labels):
    """载入nyt格式数据集
    nyt格式共6列：1、2列为两实体的freebase mid，3、4列为实体串，5列为关系名，6列为句子(以###END###结束)
        m.0ccvx    m.05gf08    queens    belle_harbor    /location/location/contains
        .....officials yesterday to reopen their investigation into the fatal crash of a passenger
        jet in belle_harbor , queens...... ###END###
    """
    with open(file, encoding='utf8') as f:
        lines = [line.strip().split() for line in f.readlines()]
    dataset = []
    for i, line in enumerate(lines):
        e1, e2, rel = line[2:5]
        sentence = line[5:]
        if sentence[-1] == '###END###':
            sentence.pop(-1)
        if len(sentence) > max_len:
            # print('exceed max length: %d/%d\n %s' % (len(sentence), max_len, sentence[:10]))
            continue
        # relation id, head and tail entity ids, and word indices
        rel_id = labels[rel]
        dataset.append((*encode(sentence, e1, e2, pos_limit, vocab), rel_id))
    return dataset


class PCNNModel(tf.keras.Model):
    """分段卷积神经网络"""

    def __init__(self, n_classes, vocab_size, pos_size, word_embedding_size=200,
                 pos_embedding_size=5, feature_maps=200, dropout_rate=0,
                 use_pretrain_vec=True, pretrained_vecs=None):
        super(PCNNModel, self).__init__()

        self.embedding_size = word_embedding_size + 2 * pos_embedding_size
        self.feature_maps = feature_maps
        self.n_classes = n_classes
        # word embedding layer
        if use_pretrain_vec:
            embeddings_initializer = tf.initializers.Constant(pretrained_vecs)
        else:
            embeddings_initializer = 'uniform'
        self.word_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=word_embedding_size,
            embeddings_initializer=embeddings_initializer,
            trainable=not use_pretrain_vec,
            name='word_embedding')
        # position embedding layer
        self.pos1_embedding = tf.keras.layers.Embedding(pos_size, pos_embedding_size, name='pos1')
        self.pos2_embedding = tf.keras.layers.Embedding(pos_size, pos_embedding_size, name='pos2')
        # 卷积核窗口视野
        self.views = [3, 4, 5]
        # convolution layers
        self.convs = [self.conv_layer(w, 'conv%d' % i) for i, w in enumerate(self.views)]
        # dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        # fully connect layer
        self.fully_connect = tf.keras.layers.Dense(self.n_classes, name='fully_connect')

    def conv_layer(self, hight=3, name=None):
        return tf.keras.layers.Conv2D(filters=self.feature_maps,
                                      kernel_size=(hight, self.embedding_size),
                                      strides=(1, self.embedding_size),
                                      padding='same',
                                      name=name)

    def piecewise_conv_maxpool(self, word_ids, pos1_ids, pos2_ids):
        """分段卷积"""
        word_embedding = self.word_embedding(word_ids)
        pos1_embedding = self.pos1_embedding(pos1_ids)
        pos2_embedding = self.pos2_embedding(pos2_ids)
        embedding = tf.concat((word_embedding, pos1_embedding, pos2_embedding), axis=-1)
        # shape=(batch_size, seq_len, embedding_size, 1)
        embedding = tf.expand_dims(embedding, -1)
        maxpools = []
        for conv in self.convs:
            # (batch_size, seq_len, 1, featue_maps)
            conv = conv(embedding)
            # (batch_size, 1, featue_maps)
            maxpool = tf.reduce_max(conv, axis=1, keepdims=False)
            # (batch_size, feature_maps, 1)
            maxpool = tf.transpose(maxpool, (0, 2, 1))
            maxpools.append(maxpool)
        # shape=(batch_size, feature_maps, windows)))
        maxpools = tf.concat(maxpools, axis=-1)
        return maxpools

    def call(self, left, middle, right, training=False):
        # 对三段子句使用相同的卷积核，卷积核数目为windows*feature_maps
        # shape=(batch_size, feature_maps, windows)
        maxpool1 = self.piecewise_conv_maxpool(*left)
        maxpool2 = self.piecewise_conv_maxpool(*middle)
        maxpool3 = self.piecewise_conv_maxpool(*right)
        # shape=(batch_size, feature_maps, views*3)
        maxpool = tf.concat((maxpool1, maxpool2, maxpool3), axis=-1)
        # shape=(batch_size, feature_maps*views*3)
        maxpool = tf.reshape(maxpool, shape=(-1, 3 * self.feature_maps * len(self.views)))
        # no-linear transform by tanh
        gvector = tf.tanh(maxpool)
        gvector = self.dropout1(gvector, training=training)
        # shape=(batch_size, n_classes)
        logits = self.fully_connect(gvector)
        return logits


class WarmupPolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """暖启+多项式衰减学习率"""

    def __init__(self, init_lr=1e-3, steps=1e8, end_lr=1e-6, warmup_steps=0., warmup_init_lr=1e-6):
        super(WarmupPolynomialDecay, self).__init__()

        self.init_lr = tf.constant(init_lr, dtype=tf.float32, name='init_learning_rate')
        self.end_lr = tf.constant(end_lr, dtype=tf.float32, name='init_learning_rate')
        # warm up steps
        self.warmup_steps = tf.constant(
            tf.maximum(float(warmup_steps), 0.), dtype=tf.float32, name='warmup_steps')
        # warm up mimimum learning rate
        self.warmup_init_lr = tf.constant(
            tf.minimum(warmup_init_lr, self.init_lr * 0.1), dtype=tf.float32, name='warmup_init')
        # learning rate
        self.leanring_rate = tf.keras.optimizers.schedules.PolynomialDecay(init_lr, steps, end_lr)

    def __call__(self, step):
        warmup = tf.cast(step < self.warmup_steps, tf.float32)
        percent = step / (self.warmup_steps + 1.)
        lr1 = self.leanring_rate(step)
        lr2 = self.warmup_init_lr + (self.init_lr - self.warmup_init_lr) * percent
        return lr1 * (1.0 - warmup) + warmup * lr2

    def get_config(self):
        return {
            'init_lr': self.init_lr,
            'end_lr': self.end_lr,
            'warmup_steps': self.warmup_steps,
            'warmup_init_lr': self.warmup_init_lr
        }


TSI = tf.TensorSpec(shape=(None, None), dtype=tf.int32)
TSO = tf.TensorSpec(shape=(None,), dtype=tf.int32)


class PCNNRelationExtraction(object):
    """PCNN关系抽取"""

    def __init__(self, vocab, labels, pos_limit=50, word_embedding_size=200,
                 pos_embedding_size=5, feature_maps=100, init_lr=1e-3, end_lr=1e-6, lr_warmup=0.1,
                 mlt_warmup=0., dropout_rate=0, output_path='./output',
                 use_pretrain_vec=True, pretrained_vecs=None):
        self.vocab = vocab
        self.labels, _ = zip(*sorted(labels.items(), key=lambda x: x[1]))
        self.n_classes = len(labels)
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.lr_warmup = lr_warmup
        self.mlt_warmup = mlt_warmup
        self.output_path = output_path
        self.pos_limit = pos_limit
        # pcnn model
        pos_size = self.pos_limit * 2 + 2
        vocab_size = len(vocab)
        self.pcnn = PCNNModel(n_classes=self.n_classes, vocab_size=vocab_size, pos_size=pos_size,
                              word_embedding_size=word_embedding_size,
                              pos_embedding_size=pos_embedding_size,
                              feature_maps=feature_maps, dropout_rate=dropout_rate,
                              use_pretrain_vec=use_pretrain_vec, pretrained_vecs=pretrained_vecs)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(True, reduction='none')

        self.n_steps = None
        self.optimizer = None
        self.mlt_warmup_steps = None

        # training metrics
        self.metric_loss = tf.keras.metrics.Mean(name='loss')
        self.metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step", trainable=False)
        self.checkpoint_manager = self.make_checkpoint_manager()

    def make_checkpoint_manager(self):
        """Checkpoint管理器"""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        checkpoint = tf.train.Checkpoint(pcnn=self.pcnn)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.output_path, max_to_keep=3, checkpoint_name='pcnn.ckpt')
        return checkpoint_manager

    def save_model(self):
        """保存模型"""
        checkpoint_path = self.checkpoint_manager.save(self.global_step)
        print('PCNN model saved: %s' % checkpoint_path)

    def restore_model(self):
        """还原模型"""
        checkpoint_path = self.checkpoint_manager.latest_checkpoint
        if not checkpoint_path:
            raise ValueError('No model exists in `%s`' % self.output_path)
        print('Restoring model from %s...' % checkpoint_path)
        self.checkpoint_manager.checkpoint.restore(checkpoint_path)

    def record_metrics(self, loss, y_true, y_pred):
        """记录评价指标"""
        self.metric_loss(loss)
        self.metric_accuracy(y_true, y_pred)

    def report_metrics(self, msg='', reset=True):
        """打印评价指标"""
        loss = self.metric_loss.result()
        acc = self.metric_accuracy.result()
        if reset:
            self.reset_metrics()
        print('{}, loss:{:.3f}, accuracy:{:.3f}'.format(msg, loss, acc).strip(', '))

    def reset_metrics(self):
        """重置评价指标累计结果"""
        self.metric_loss.reset_states()
        self.metric_accuracy.reset_states()

    def piecewise_split(self, word_ids, ent_ids, pos1_ids, pos2_ids):
        """按实体位置分段切割数据集为三部分"""
        piecewise1, piecewise2, piecewise3 = [], [], []
        for words, poss1, poss2, (i, j) in zip(word_ids, pos1_ids, pos2_ids, ent_ids):
            if i > j:
                i, j = j, i
            piecewise1.append((words[:i + 1], poss1[:i + 1], poss2[:i + 1]))
            piecewise2.append((words[i:j + 1], poss1[i:j + 1], poss2[i:j + 1]))
            piecewise3.append((words[j:], poss1[j:], poss2[j:]))
        piecewise1 = [self.pad_sequences(d) for d in zip(*piecewise1)]
        piecewise2 = [self.pad_sequences(d) for d in zip(*piecewise2)]
        piecewise3 = [self.pad_sequences(d) for d in zip(*piecewise3)]
        # left, middle and right dataset
        return piecewise1, piecewise2, piecewise3

    @staticmethod
    def pad_sequences(sequence, max_len=None, value=0):
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequence, max_len, dtype='int32', padding='post', truncating='post', value=value)

    def loss_function(self, y_true, y_pred):
        """损失函数"""
        batch_loss = tf.reduce_mean(self.loss_object(y_true, y_pred))
        # 多实例学习, global step从0开始，共执行mlt_warmup_steps次非多实例学习
        if self.global_step >= self.mlt_warmup_steps:
            # 仅计算每个包（类别）输出真实标签概率最大的实例的损失
            mask = tf.one_hot(y_true, depth=self.n_classes, dtype=tf.float32)
            mlt_ids = tf.cast(tf.argmax(y_pred * mask, axis=0), dtype=tf.int32)
            # 掩码不含实例的空包
            mlt_ids *= tf.cast(tf.equal(tf.reduce_max(mask, axis=0), 1), dtype=tf.int32)
            y_pred = tf.gather(y_pred, mlt_ids)
            y_true = tf.gather(y_true, mlt_ids)
            mlt_loss = tf.reduce_mean(self.loss_object(y_true, y_pred))
        else:
            mlt_loss = batch_loss
        return mlt_loss, batch_loss

    def train(self, train_dataset, eval_dataset, epochs, batch_size=64):
        """模型训练"""
        n_examples = len(train_dataset)
        n_steps = epochs * math.ceil(n_examples / batch_size)
        print('total examples: %d, batch_size: %d, epochs: %d, steps: %d' % (
            n_examples, batch_size, epochs, n_steps))

        self.n_steps = tf.constant(n_steps, dtype=tf.int32, name='n_steps')
        self.mlt_warmup_steps = tf.constant(
            int(n_steps * self.mlt_warmup), dtype=tf.int32, name='warmup_steps')
        # adam optimizer
        lr_warmup_steps = int(self.lr_warmup * n_steps)
        lr = WarmupPolynomialDecay(
            init_lr=self.init_lr, steps=n_steps, end_lr=self.end_lr, warmup_steps=lr_warmup_steps)
        self.optimizer = tf.keras.optimizers.Adam(lr)

        for epoch in range(epochs):
            print('\nEpoch %d' % epoch)
            # shuffle dataset
            np.random.shuffle(train_dataset)
            for i in range(0, n_examples, batch_size):
                words, ents, poss1, poss2, rels = zip(*train_dataset[i:i + batch_size])
                # piecewise dataset
                left, middle, right = self.piecewise_split(words, ents, poss1, poss2)
                self.train_step(left, middle, right, rels)
                # report traing process
                if self.global_step % 100 == 0:
                    step = self.global_step.numpy()
                    self.report_metrics(msg='step %d' % step)
                # checkpoint
                # if self.global_step % 1000 == 0:
                #     self.save_model()
            # evaluate performance
            # self.evaluate(eval_dataset)
        # checkpoint
        self.save_model()

    @tf.function(input_signature=[[TSI, TSI, TSI], [TSI, TSI, TSI], [TSI, TSI, TSI], TSO])
    def train_step(self, left, middle, right, rel_ids):
        """执行一次梯度更新"""
        # multi-instance learning warmup
        with tf.GradientTape() as tape:
            logits = self.pcnn(left, middle, right, training=True)
            mlt_loss, batch_loss = self.loss_function(rel_ids, logits)
        gradients = tape.gradient(batch_loss, self.pcnn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.pcnn.trainable_variables))
        # statistic metrics
        self.record_metrics(batch_loss, rel_ids, logits)
        # increasing global step
        self.global_step.assign_add(1)

    def evaluate(self, dataset, batch_size=32):
        """模型评估"""
        self.reset_metrics()
        for i in range(0, len(dataset), batch_size):
            words, ents, poss1, poss2, rels = zip(*dataset[i:i + batch_size])
            # piecewise dataset
            left, middle, right = self.piecewise_split(words, ents, poss1, poss2)
            logits = self.pcnn(left, middle, right, training=False)
            _, batch_loss = self.loss_function(rels, logits)
            self.record_metrics(batch_loss, rels, logits)

        self.report_metrics(msg='evaluation %d examples' % len(dataset))

    def predict(self, inputs, targets=None, batch_size=32):
        """模型预测
        inputs: [[sentence, entity1, entity2], ...]
        """
        # restore model
        self.restore_model()
        dataset = [encode(sent, e1, e2, self.pos_limit, self.vocab) for sent, e1, e2 in inputs]
        result = []
        for i in range(0, len(dataset), batch_size):
            words, ents, poss1, poss2 = zip(*dataset[i:i + batch_size])
            left, middle, right = self.piecewise_split(words, ents, poss1, poss2)
            logits = self.pcnn(left, middle, right, training=False)
            probs = tf.nn.softmax(logits, axis=1).numpy()
            indices = np.argmax(probs, axis=1)
            preds = [self.labels[i] for i in indices]
            probs = probs[range(probs.shape[0]), indices]
            result.extend((pred, prob) for pred, prob in zip(preds, probs))

        for i, (pred, prob) in enumerate(result):
            if i == 20:
                break
            string = '\n%s\n  prediction/probability:%s/%.3f' % (inputs[i], pred, prob)
            if targets is not None:
                string += ', target:%s' % targets[i]
            print(string)
        return result


def main():
    """主函数入口"""
    if args.do_train:
        build_corpus(corpus_file=os.path.join(args.data_path, 'new-train.txt'),
                     vocab_path=args.vocab_path,
                     label_path=args.label_path)
    if args.use_pretrained_vec:
        with open(os.path.join(args.data_path, 'words.txt'), encoding='utf8') as f:
            words = [line.strip() for line in f.readlines()]
            vocab = dict(zip(words, range(len(words))))
        pickle.dump(vocab, open(args.vocab_path, 'wb'))
        vectors = np.load(os.path.join(args.data_path, 'vec.npy'))
        assert args.word_embedding_size == vectors.shape[1]
    elif args.use_glove_vec:
        # load glove word vector
        glove_path = args.glove_path
        if not os.path.isfile(args.glove_path):
            glove_path = tf.keras.utils.get_file(
                'glove.6B.zip', origin='http://nlp.stanford.edu/data/glove.6B.zip', extract=True)
            glove_path = os.path.join(os.path.dirname(glove_path), 'glove.6B.200d.txt')
            # cp /root/.keras/datasets/glove.6B.200d.txt "/content/drive/My Drive/model/glove"
        tmp_file = get_tmpfile("glove.6B.200d.word2vec.txt")
        glove2word2vec(glove_path, tmp_file)
        print('Loadding pre-trained glove vector...')
        glove_vec = KeyedVectors.load_word2vec_format(tmp_file)
        # glove vocabulary
        vocab = dict({'<PAD>': 0, '<UNK>': 1}, **dict(
            zip(glove_vec.vocab.keys(), range(2, len(glove_vec.vocab) + 2))))
        assert args.word_embedding_size == glove_vec.vectors.shape[1]
        # glove vectors
        vectors = np.concatenate(
            (np.random.randn(2, args.word_embedding_size), glove_vec.vectors), axis=0)
        pickle.dump(vocab, open(args.vocab_path, 'wb'))
        del glove_vec
    else:
        vectors = None

    vocab = pickle.load(open(args.vocab_path, 'rb'))
    labels = pickle.load(open(args.label_path, 'rb'))
    print('labels: ', labels)

    model = PCNNRelationExtraction(
        labels=labels,
        vocab=vocab,
        pos_limit=args.pos_limit,
        word_embedding_size=args.word_embedding_size,
        pos_embedding_size=args.pos_embedding_size,
        feature_maps=args.feature_maps,
        init_lr=args.init_lr,
        end_lr=args.end_lr,
        lr_warmup=args.lr_warmup,
        mlt_warmup=args.mlt_warmup,
        dropout_rate=args.dropout_rate,
        output_path=args.output_path,
        use_pretrain_vec=args.use_pretrained_vec or args.use_glove_vec,
        pretrained_vecs=vectors
    )

    if args.do_train:
        train_dataset = load_nyt_dataset(
            os.path.join(args.data_path, 'new-train.txt'), max_len=args.max_len,
            pos_limit=args.pos_limit, vocab=vocab, labels=labels)
        test_dataset = load_nyt_dataset(
            os.path.join(args.data_path, 'new-test.txt'), max_len=args.max_len,
            pos_limit=args.pos_limit, vocab=vocab, labels=labels)
        model.train(train_dataset, test_dataset, epochs=args.epochs, batch_size=args.batch_size)

    if args.do_predict:
        inputs, targets = [], []
        with open(os.path.join(args.data_path, 'new-test.txt'), encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip().split()
                inputs.append((line[5:], line[2], line[3]))
                targets.append(line[4])
        result = model.predict(inputs, targets)


if __name__ == '__main__':
    main()
