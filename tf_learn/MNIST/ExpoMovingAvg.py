import tensorflow as tf
v1 = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(0, trainable=False)

# this define the class of moving averages, which is ExponentialMA
ema = tf.train.ExponentialMovingAverage(0.99, step)

# this will update the shadow variable to the value of movingaverage
# also the dacay rate will be updated too
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("maintain_averagers_op: ", sess.run(maintain_averages_op))

    print(sess.run([v1, ema.average(v1)]))
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)
    print("maintain_averagers_op: ", sess.run(maintain_averages_op))
    # ema.average(v1) will show value of the shadow variable of v1
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print("maintain_averagers_op: ", sess.run(maintain_averages_op))
    print(sess.run([v1, ema.average(v1)]))


    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))