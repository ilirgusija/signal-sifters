# %% [code]
#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import json
import os

DATASET_DIR = "../data"

TRAINING_SET_LABELED = [
    {
        "tfrecords" : os.path.join(DATASET_DIR, "dichasus-cf02.tfrecords"),
        "offsets" : os.path.join(DATASET_DIR, "reftx-offsets-dichasus-cf02.json")
    }
]

TRAINING_SET_UNLABELED = [
    {
        "tfrecords" : os.path.join(DATASET_DIR, "dichasus-cf04.tfrecords"),
        "offsets" : os.path.join(DATASET_DIR, "reftx-offsets-dichasus-cf04.json")
    },
    {
        "tfrecords" : os.path.join(DATASET_DIR, "dichasus-cf05.tfrecords"),
        "offsets" : os.path.join(DATASET_DIR, "reftx-offsets-dichasus-cf05.json")
    }
]

TEST_SET = [
    {
        "tfrecords" : os.path.join(DATASET_DIR, "dichasus-cf03.tfrecords"),
        "offsets" : os.path.join(DATASET_DIR, "reftx-offsets-dichasus-cf03.json")
    }
]

RAW_SUBCARRIER_COUNT = 1024

antenna_count = 0
array_positions = []
array_normalvectors = []
array_upvectors = []
array_rightvectors = []
antenna_assignments = []

with open(os.path.join(DATASET_DIR, "spec.json")) as specfile:
    spec = json.load(specfile)
    for antenna in spec["antennas"]:
        antenna_count = antenna_count + sum([len(row) for row in antenna["assignments"]])
        antenna_assignments.append(antenna["assignments"])
        
        array_positions.append(np.asarray(antenna["location"]))
        array_upvectors.append(np.asarray(antenna["upvector"]))
        array_rightvectors.append(np.asarray(antenna["rightvector"]))

        normalvector = np.cross(np.asarray(antenna["rightvector"]), np.asarray(antenna["upvector"]))
        normalvector = normalvector / np.linalg.norm(normalvector)
        array_normalvectors.append(normalvector)

def parse_and_calibrate(path, offset_path):
    offsets = None
    with open(offset_path, "r") as offsetfile:
        offsets = json.load(offsetfile)

    def record_parse_function(proto):
        record = tf.io.parse_single_example(
            proto,
            {
                "csi": tf.io.FixedLenFeature([], tf.string, default_value = ""),
                "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value = ""),
                "time": tf.io.FixedLenFeature([], tf.float32, default_value = 0),
            },
        )

        # Normalize sampling time offset for CSI
        csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (antenna_count, RAW_SUBCARRIER_COUNT, 2))
        csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
        csi = tf.signal.fftshift(csi, axes = 1)
        incr = tf.cast(tf.math.angle(tf.math.reduce_sum(csi[:,1:] * tf.math.conj(csi[:,:-1]))), tf.complex64)
        csi = csi * tf.exp(-1.0j * incr * tf.cast(tf.range(csi.shape[-1]), tf.complex64))[tf.newaxis,:]

        position = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3))
        time = tf.ensure_shape(record["time"], ())

        return csi, position, time

    def apply_calibration(csi, pos, time):
        sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
        cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
        csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))

        return csi, pos, time

    def shrink_csi(target_subcarrier_count = 64):
        def compression(csi, pos, time):
            csi_tdomain = tf.signal.fftshift(tf.signal.ifft(tf.signal.fftshift(csi, axes = -1)), axes = -1)
            csi_tdomain_compressed = csi_tdomain[...,RAW_SUBCARRIER_COUNT // 2 - target_subcarrier_count // 2:RAW_SUBCARRIER_COUNT // 2 + target_subcarrier_count // 2]
            return tf.signal.fftshift(tf.signal.fft(tf.signal.fftshift(csi_tdomain_compressed, axes = -1)), axes = -1), pos, time

        return compression

    def order_by_antenna_assignments(csi, pos, time):
        csi = tf.stack([[tf.gather(csi, antenna_indices) for antenna_indices in array] for array in antenna_assignments])
        return csi, pos, time

    dset = tf.data.TFRecordDataset(path)
    dset = dset.map(record_parse_function)
    dset = dset.map(apply_calibration)
    dset = dset.map(shrink_csi())
    dset = dset.map(order_by_antenna_assignments)

    return dset

def load_datasets(paths_list):
    full_dataset = parse_and_calibrate(paths_list[0]["tfrecords"], paths_list[0]["offsets"])

    for path in paths_list[1:]:
        full_dataset = full_dataset.concatenate(parse_and_calibrate(path["tfrecords"], path["offsets"]))

    return full_dataset.shard(4, 0)

# You may access the following variables for training
array_positions = np.asarray(array_positions)
array_normalvectors = np.asarray(array_normalvectors)
array_upvectors = np.asarray(array_upvectors)
array_rightvectors = np.asarray(array_rightvectors)
training_set_labeled = load_datasets(TRAINING_SET_LABELED)
training_set_unlabeled = load_datasets(TRAINING_SET_UNLABELED).map(lambda csi, pos, time : (csi, time))

# You may access the following variables for evaluation, but not for training
test_set = load_datasets(TEST_SET)
