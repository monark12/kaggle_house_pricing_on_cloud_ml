import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import models
from tensorflow.keras.optimizers import Adam
tf.logging.set_verbosity(tf.logging.INFO)

def keras_estimator(model_dir, config, params):
    model = models.Sequential()
    model.add(Dense(units=480, kernel_initializer='random_uniform', activation= 'relu',
    			input_shape=(params['num_features'],)))
    model.add(Dense(units=480, kernel_initializer='random_uniform', activation= 'relu'))
    model.add(Dense(units=10, kernel_initializer='random_uniform', activation= 'relu'))
    model.add(Dense(units=1, kernel_initializer='random_uniform', activation= 'elu'))
    optimizer = Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['mse'])
    return tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir, 
    												config=config)

def input_fn(features, labels, batch_size, mode):
	if labels is None:
		inputs = features
	else:
		inputs = (features, labels)
	dataset = tf.data.Dataset.from_tensor_slices(inputs)
	if mode == tf.estimator.ModeKeys.TRAIN:
		dataset = dataset.shuffle(1000).repeat().batch(batch_size)
	if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
		dataset = dataset.batch(batch_size)
	return dataset.make_one_shot_iterator().get_next()

def serving_input_fn():
	feature_placeholder = tf.placeholder(tf.float32, [None, 304])
	features = feature_placeholder
	return tf.estimator.export.TensorServingInputReceiver(features, feature_placeholder)
