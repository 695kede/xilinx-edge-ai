'''
Helper script to import a protobuf graph into Tensorboard

Example usage:
python import_pb.py --graph=./freeze/frozen_graph.pb
'''

import argparse
import os
import shutil
from tensorflow.python.tools import import_pb_to_tensorboard


def run_main():
  parser = argparse.ArgumentParser(description='Script to import frozen graph into TensorBoard')
  parser.add_argument(
      '-g', '--graph',
      type=str,
      default='',
      help="Protobuf graph file (.pb) to import into TensorBoard.",
      required='True')

  flags = parser.parse_args()

  if (os.path.exists('./dummy')):
    shutil.rmtree('./dummy')

  import_pb_to_tensorboard.import_to_tensorboard(model_dir=flags.graph, log_dir='./dummy')
  print('..or try `tensorboard --logdir=./dummy --port 6006 --host localhost`')


if __name__ == '__main__':
    run_main()

