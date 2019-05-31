import tensorflow as tf

def miniVGGNet(inputs, is_training, drop_rate):
    net = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(inputs=net, training=is_training)
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.dropout(inputs=net, rate=drop_rate, training=is_training)

    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(inputs=net, training=is_training)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(inputs=net, training=is_training)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.dropout(inputs=net, rate=drop_rate, training=is_training)

    net = tf.layers.flatten(inputs=net)
    net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(inputs=net, training=is_training)
    net = tf.layers.dropout(inputs=net, rate=drop_rate, training=is_training)
    logits = tf.layers.dense(inputs=net, units=10, activation=None)
    return logits


