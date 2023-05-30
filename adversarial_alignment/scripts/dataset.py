import tensorflow as tf

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.AUTO = tf.data.AUTOTUNE

        self._feature_description = {
            "image"       : tf.io.FixedLenFeature([], tf.string, default_value=''),
            "heatmap"     : tf.io.FixedLenFeature([], tf.string, default_value=''),
            "label"       : tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }

    def parse_prototype(self, prototype, training=False):
        data    = tf.io.parse_single_example(prototype, self._feature_description)

        image   = tf.io.decode_raw(data['image'], tf.float32)
        image   = tf.reshape(image, (224, 224, 3))
        image   = tf.cast(image, tf.float32)

        heatmap = tf.io.decode_raw(data['heatmap'], tf.float32)
        heatmap = tf.reshape(heatmap, (224, 224, 1))

        label   = tf.cast(data['label'], tf.int32)
        label   = tf.one_hot(label, 1_000)

        return image, heatmap, label

    def get_dataset(self, batch_size, training=False):
        deterministic_order = tf.data.Options()
        deterministic_order.experimental_deterministic = True

        dataset = tf.data.TFRecordDataset([self.data_path], num_parallel_reads=self.AUTO)
        dataset = dataset.with_options(deterministic_order) 
        
        dataset = dataset.map(self.parse_prototype, num_parallel_calls=self.AUTO)
        
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.AUTO)

        return dataset