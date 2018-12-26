import tensorflow as tf
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
    [lstm_cell(lstm_size) for _ in range(number_of_layers)]
)
state= stacked_lstm.zero_state(batch_size, tf.float32)

for i in range(len(num_steps)):
    if i >0: tf.get_variable_scope().reuse_variables()
    stacked_lstm_output,state=stacked_lstm(current_input, state)
    final_output=fully_connected(stacked_lstm_output)
    loss+=calc_loss(final_output, expected_output)