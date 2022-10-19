#-------------------------------------------------------------------------------
# Name:        SensorID5127
# Purpose:

# Created:     12-12-2019
# Copyright:   (c) t-82maca1mpg 2019
##special tahnks to Weimin Wang (https://weiminwang.blog/2017/09/29/multivariate-time-series-forecast-using-seq2seq-in-tensorflow/)
##Modified by   Caleb Juma

#-------------------------------------------------------------------------------

# Imports

# DL libraries
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes

# Data Manipulation
import numpy as np
import pandas as pd
import random
#import math

# Files/OS
import os
import copy

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Benchmarking
import time

# Error Analysis
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


plt.rcParams.update({'font.size':23})
##np.random.seed(42)
##tf.random.set_random_seed(42)

input_file=r'D:\PROJECTS\Humburg_Lake_Victoria\data\Winam_Gulf_Satellite_Data.csv'

df_original = pd.read_csv(input_file)#put in a diff folder to the py file
print(df_original.head())

#df=df_original.loc[(df_original != 0).all(1)]
df=df_original[(df_original["mean_monthly_temp"] > 0)]




#visualize the dataset#"Year","month",
cols_to_plot = ["Year","month", "mean_monthly_temp",\
 'mean_monthly_rainfall', "littoral_mean_monthly_ndvi","littoral_vegetation_area","mean_monthly_ndbi","Bare_soil_area","mean_monthly_ndwi","water_area","hyacinth_mean_monthly_ndvi","hyacinth_area"]


## Split into train and test - Ithe month of Nov was used as the test datasource
df_train = df.iloc[:(-50), :].copy()# -256 refers to counting 256 from the last record. Thus from first record to (end-256)
df_test = df.iloc[-50:, :].copy()


#"Year","month",
## take out the useful columns for modeling - you may also keep 'hour', 'day' or 'month' and to see if that will improve your accuracy
X_train = df_train.loc[:, ["Year","month","mean_monthly_temp","mean_monthly_rainfall","littoral_vegetation_area","water_area","Bare_soil_area","mean_monthly_ndbi","mean_monthly_ndwi","littoral_mean_monthly_ndvi"]].values.copy()
X_test = df_test.loc[:, ["Year","month","mean_monthly_temp","mean_monthly_rainfall","littoral_vegetation_area","water_area","Bare_soil_area","mean_monthly_ndbi","mean_monthly_ndwi","littoral_mean_monthly_ndvi"]].values.copy()
y_train = df_train["hyacinth_area"].values.copy().reshape(-1, 1)
y_test = df_test["hyacinth_area"].values.copy().reshape(-1, 1)


## z-score transform x -to ensure the averages are computed from the same base
for i in range(X_train.shape[1]):
    temp_mean = X_train[:, i].mean()
    temp_std = X_train[:, i].std()
    X_train[:, i] = (X_train[:, i] - temp_mean) / temp_std
    X_test[:, i] = (X_test[:, i] - temp_mean) / temp_std

## z-score transform y
y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

input_seq_len =24
output_seq_len =1

#np.random.seed(1) # to reproduce the randomness

#transform into 3D format for timeseries ie having batch size, time_steps, and feature dimension
def generate_train_samples(x = X_train, y = y_train, batch_size = 4,\
                          input_seq_len = input_seq_len, output_seq_len = output_seq_len):

    total_start_points = len(x) - input_seq_len - output_seq_len
    print("2nd lenx=", len(x))
    print ("total start point =",total_start_points)

##    np.random.seed(42)

    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)

    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)

    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)

    return input_seq, output_seq # in shape: (batch_size, time_steps, feature_dim)



def generate_test_samples(x = X_test, y = y_test, input_seq_len = input_seq_len, output_seq_len = output_seq_len):

    total_samples = x.shape[0]

    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)

    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)

    return input_seq, output_seq


x, y = generate_train_samples()
print(x.shape, y.shape)

test_x, test_y = generate_test_samples()
print("test samples",test_x.shape, test_y.shape)


#Train the model
## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003

## Network Parameters
# length of input signals
input_seq_len = input_seq_len
# length of output signals
output_seq_len = output_seq_len
# size of LSTM Cell
hidden_dim =14
# num of input signals
input_dim = X_train.shape[1]
# num of output signals
output_dim = y_train.shape[1]
# num of stacked lstm layers
num_stacked_layers = 2
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 1.5






def build_graph(feed_previous = False):

    tf.reset_default_graph()

    global_step = tf.Variable(
                  initial_value=0,
                  name="global_step",
                  trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    weights = {
        'out': tf.get_variable('Weights_out', \
                               shape = [hidden_dim, output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.truncated_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out', \
                               shape = [output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.constant_initializer(0.)),
    }

    with tf.variable_scope('Seq2seq'):
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
               for t in range(input_seq_len)
        ]

        # Decoder: target outputs
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
              for t in range(output_seq_len)
        ]

        # Give a "GO" token to the decoder.
        # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
        # first element will be fed as decoder input which is then 'un-guided'
        dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

        with tf.variable_scope('LSTMCell'):
            cells = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        def _rnn_decoder(decoder_inputs,
                        initial_state,
                        cell,
                        loop_function=None,
                        scope=None):
          """RNN decoder for the sequence-to-sequence model.
          Args:
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            initial_state: 2D Tensor with shape [batch_size x cell.state_size].
            cell: rnn_cell.RNNCell defining the cell function and size.
            loop_function: If not None, this function will be applied to the i-th output
              in order to generate the i+1-st input, and decoder_inputs will be ignored,
              except for the first element ("GO" symbol). This can be used for decoding,
              but also for training to emulate http://arxiv.org/abs/1506.03099.
              Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
            scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing generated outputs.
              state: The state of each cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
                (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                 states can be the same. They are different for LSTM cells though.)
          """
          with variable_scope.variable_scope(scope or "rnn_decoder"):
            state = initial_state
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):
              if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                  inp = loop_function(prev, i)
              if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
              output, state = cell(inp, state)
              outputs.append(output)
              if loop_function is not None:
                prev = output
          return outputs, state

        def _basic_rnn_seq2seq(encoder_inputs,
                              decoder_inputs,
                              cell,
                              feed_previous,
                              dtype=dtypes.float32,
                              scope=None):
          """Basic RNN sequence-to-sequence model.
          This model first runs an RNN to encode encoder_inputs into a state vector,
          then runs decoder, initialized with the last encoder state, on decoder_inputs.
          Encoder and decoder use the same RNN cell type, but don't share parameters.
          Args:
            encoder_inputs: A list of 2D Tensors [batch_size x input_size].
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            feed_previous: Boolean; if True, only the first of decoder_inputs will be
              used (the "GO" symbol), all other inputs will be generated by the previous
              decoder output using _loop_function below. If False, decoder_inputs are used
              as given (the standard decoder case).
            dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
            scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
              state: The state of each decoder cell in the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
          """
          with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
            enc_cell = copy.deepcopy(cell)
            _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
            if feed_previous:
                return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
            else:
                return _rnn_decoder(decoder_inputs, enc_state, cell)

        def _loop_function(prev, _):
          '''Naive implementation of loop function for _rnn_decoder. Transform prev from
          dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
          used as decoder input of next time step '''
          return tf.matmul(prev, weights['out']) + biases['out']

        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            enc_inp,
            dec_inp,
            cell,
            feed_previous = feed_previous
        )

        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=learning_rate,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=GRADIENT_CLIPPING)

    saver = tf.train.Saver

    return dict(
        enc_inp = enc_inp,
        target_seq = target_seq,
        train_op = optimizer,
        loss=loss,
        saver = saver,
        reshaped_outputs = reshaped_outputs,
        )


# ## Train


epochs = 10000
batch_size = 100# if not specified, it will use batch size specified in the method!!
KEEP_RATE = 0.7 #0.7
train_losses = []
val_losses = []



rnn_model = build_graph(feed_previous=False)

saver = tf.train.Saver()

start_train = time.time()
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    print("Training losses: ")
    for i in range(epochs):
        batch_input, batch_output = generate_train_samples(batch_size=batch_size)

        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print("loss value is =",loss_t)

    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./', 'multivariate_ts_hyacinth_case'))

print("Checkpoint saved at: ", save_path)


# #### Benchmark time it takes to train
end_train = time.time()


print('Time taken to train is {} minutes'.format((end_train - start_train) / 60))
print('Time in seconds is: {}'.format(end_train - start_train))
train_time = end_train - start_train


# ## Inference on test
start_test = time.time()

rnn_model = build_graph(feed_previous=True)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    saver = rnn_model['saver']().restore(sess,  os.path.join('./', 'multivariate_ts_hyacinth_case'))

    feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

    final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
    final_preds = np.concatenate(final_preds, axis = 1)
    print("Test mse is: ", np.mean((final_preds - test_y)**2))


# #### Benchmark time it takes to test

end_test = time.time()


print('Time taken is: {} minutes.'.format((end_test - start_test) / 60))
print('Time in seconds is: {}'.format(end_test - start_test))



test_time = end_test - start_test



##
##
# ### Unscale the predictions'''i.e Remove the z factor that had been used earlier
# make into one array
dim1, dim2 = final_preds.shape[0], final_preds.shape[1]

preds_flattened = final_preds.reshape(dim1*dim2, 1)
unscaled_yhat = pd.DataFrame(preds_flattened, columns=["hyacinth_area"]).apply(lambda x: (x*y_std) + y_mean)
yhat_inv = unscaled_yhat.values

test_y_flattened = test_y.reshape(dim1*dim2, 1)
unscaled_y = pd.DataFrame(test_y_flattened, columns=["hyacinth_area"]).apply(lambda x: (x*y_std) + y_mean)
y_inv = unscaled_y.values

pd.concat((unscaled_y,unscaled_yhat),axis=1)

# ### Calculate RMSE and Plot
mse = np.mean((yhat_inv - y_inv)**2)
rmse = np.sqrt(mse)



print('Root Mean Squared Error: {:.4f}'.format(rmse))


###plot the graph
def plot_test(final_preds, test_y):
    fig, ax = plt.subplots(figsize=(17,8))
    ax.set_title("hyacinth area (km² ) Predictions vs. Actual For year")
    ax.plot(final_preds, color = 'red', label = 'predicted')
    ax.plot(test_y, color = 'green', label = 'actual')
    plt.xlabel("months")
    plt.ylabel("hyacinth area in km²")
    plt.legend(loc="upper left")
    plt.show()




plot_test(yhat_inv[:50,], y_inv[:50,])

