import argparse
from . import model
import numpy as np
import tensorflow as tf
import os
import subprocess
from sklearn.preprocessing import PowerTransformer, StandardScaler
import pandas as pd


WORKING_DIR = os.getcwd()
TEMP_DIR = 'tmp/'
DATA_FILE_NAME = 'kaggle_housing_prices.csv'


def download_files_from_gcs(source, destination):
    local_file_names = [destination]
    gcs_input_paths = [source]
    
    raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name) for local_file_name in local_file_names]
    for i, gcs_input_path in enumerate(gcs_input_paths):
        if gcs_input_path:
            subprocess.check_call(['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])
    
    return raw_local_files_data_paths
    

def load_data(path='kaggle_housing_prices.csv', test_split=0.2, seed=113):
    assert 0 <= test_split < 1
    if not path:
        raise ValueError('No dataset file defined')

    if path.startswith('gs://'):
        download_files_from_gcs(path, destination=DATA_FILE_NAME)
        path = DATA_FILE_NAME

    df_train = pd.read_csv(path)
    df_train = df_train[df_train.columns.difference(['Id'])]
    df_train = df_train.fillna(0)
    df_train = pd.get_dummies(df_train)
    X = df_train[df_train.columns.difference(['SalePrice'])].values
    y = df_train[['SalePrice']].values
    pt_X = PowerTransformer(method='yeo-johnson', standardize=False)
    sc_y = StandardScaler()
    sc_X = StandardScaler()
    y = sc_y.fit_transform(y)
    X = sc_X.fit_transform(X)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    x_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(y[:int(len(X) * (1 - test_split))])
    x_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(y[int(len(X) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--job-dir',
    	type=str,
    	help='GCS location to write checkpoints and export models')
    parser.add_argument(
    	'--train-file',
    	type=str,
    	required=True,
    	help='Dataset file local or GCS')
    parser.add_argument(
    	'--test-split',
    	type=float,
    	default=0.2,
    	help='Split between training and test, default=0.2')
    parser.add_argument(
    	'--num-epochs',
    	type=float,
    	default=500,
    	help='number of times to go through the data, default=500')
    parser.add_argument(
    	'--batch-size',
    	type=int,
    	default=128,
    	help='number of records to read during each training step, default=128')
    parser.add_argument(
    	'--learning-rate',
    	type=float,
    	default=.001,
    	help='learning rate for gradient descent, default=.001')
    parser.add_argument(
    	'--verbosity',
    	choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    	default='INFO')
    args, _ = parser.parse_known_args()
    return args

def train_and_evaluate(args):
    (train_data,train_labels), (test_data,test_labels) = load_data(path=args.train_file)
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
    train_steps = args.num_epochs * len(train_data) / args.batch_size
    train_labels = np.asarray(train_labels).astype('float32').reshape((-1, 1))
    train_spec = tf.estimator.TrainSpec(
    	input_fn=lambda: model.input_fn(
    		train_data,
    		train_labels,
    		args.batch_size,
    		mode=tf.estimator.ModeKeys.TRAIN),
    	max_steps=train_steps)
    exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
    test_labels = np.asarray(test_labels).astype('float32').reshape((-1, 1))
    eval_spec = tf.estimator.EvalSpec(
    	input_fn=lambda: model.input_fn(
    		test_data,
    		test_labels,
    		args.batch_size,
    		mode=tf.estimator.ModeKeys.EVAL),
    	steps=None,
    	exporters=[exporter],
    	start_delay_secs=10,
    	throttle_secs=10)
    estimator = model.keras_estimator(
    	model_dir=args.job_dir,
    	config=run_config,
    	params={'learning_rate': args.learning_rate,'num_features': train_data.shape[1]})
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)