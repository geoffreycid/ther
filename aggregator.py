# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""Aggregates multiple tensorbaord runs"""

import ast
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event

FOLDER_NAME = "aggregates"


def extract(dpath, subpath):
    scalar_accumulators = [EventAccumulator(str(dpath / dname / subpath)).Reload(
    ).scalars for dname in next(os.walk(dpath))[1] if dname != FOLDER_NAME]


    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    #keys = all_keys[0]
    keys = []
    for key in all_keys[0]:
        if key != "Accuracy":
            keys.append(key)
    keys = tuple(keys)
    all_scalar_events_per_key = [[scalar_accumulator.Items(key)[:5500] for scalar_accumulator in scalar_accumulators] for key in keys[:2]]
    all_scalar_events_per_key += [[scalar_accumulator.Items(key)[:1100] for scalar_accumulator in scalar_accumulators] for key in keys[2:]]
    # Get and validate all steps per key
    all_steps_per_key = [[tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get and average wall times per step per key
    wall_times_per_key = [np.mean([tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events], axis=0)
                          for all_scalar_events in all_scalar_events_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    all_per_key = dict(zip(keys, zip(steps_per_key, wall_times_per_key, values_per_key)))

    return all_per_key


def aggregate_to_summary(dpath, aggregation_ops, extracts_per_subpath):
    for op in aggregation_ops:
        for subpath, all_per_key in extracts_per_subpath.items():
            path = dpath / FOLDER_NAME / op.__name__ / subpath
            aggregations_per_key = {key: (steps, wall_times, op(values, axis=0)) for key, (steps, wall_times, values) in all_per_key.items()}
            write_summary(path, aggregations_per_key)


def write_summary(dpath, aggregations_per_key):
    writer = tf.summary.FileWriter(dpath)

    for key, (steps, wall_times, aggregations) in aggregations_per_key.items():
        for step, wall_time, aggregation in zip(steps, wall_times, aggregations):
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=aggregation)])
            scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
            writer.add_event(scalar_event)

        writer.flush()


def aggregate_to_csv(dpath, aggregation_ops, extracts_per_subpath):
    for subpath, all_per_key in extracts_per_subpath.items():
        for key, (steps, wall_times, values) in all_per_key.items():
            aggregations = [op(values, axis=0) for op in aggregation_ops]
            write_csv(dpath, subpath, key, dpath.name, aggregations, steps, aggregation_ops)


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def write_csv(dpath, subpath, key, fname, aggregations, steps, aggregation_ops):
    path = dpath / FOLDER_NAME

    if not path.exists():
        os.makedirs(path)

    file_name = get_valid_filename(key) + '-' + get_valid_filename(subpath) + '-' + fname + '.csv'
    aggregation_ops_names = [aggregation_op.__name__ for aggregation_op in aggregation_ops]
    df = pd.DataFrame(np.transpose(aggregations), index=steps, columns=aggregation_ops_names)
    df.to_csv(path / file_name, sep=';')


def aggregate(dpath, output, subpaths):
    name = dpath.name

    aggregation_ops = [np.mean, np.min, np.max, np.median, np.std, np.var]

    ops = {
        'summary': aggregate_to_summary,
        'csv': aggregate_to_csv
    }

    print("Started aggregation {}".format(name))

    extracts_per_subpath = {subpath: extract(dpath, subpath) for subpath in subpaths}

    ops.get(output)(dpath, aggregation_ops, extracts_per_subpath)

    print("Ended aggregation {}".format(name))


def param_list(param):
    p_list = ast.literal_eval(param)
    if type(p_list) is not list:
        raise argparse.ArgumentTypeError("Parameter {} is not a list".format(param))
    return p_list


def wrapper(path, output="summary"):
    path_run = path + "/" + next(os.walk(path))[1][0] + "/" #"/run_1/"
    path = Path(path)

    if not path.exists():
        raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(path))

    #subpaths = [path / dname / subpath for subpath in args.subpaths for dname in os.listdir(path) if dname != FOLDER_NAME]

    subpaths = [path / dname / subpath for subpath in next(os.walk(path_run))[1] for dname in next(os.walk(path))[1] if dname != "aggregates"]

    for subpath in subpaths:
        assert os.path.exists(subpath), "Parameter {} is not a valid path".format(subpath)

    assert output in ['summary', 'csv'], "Parameter {} is not summary or csv"

    name_subpaths = [subpath for subpath in next(os.walk(path_run))[1]]

    aggregate(path, output, name_subpaths)
