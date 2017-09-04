import tensorflow as tf


target = tf.placeholder(float, [None, 1])

def cox_cummulative(target, time):
    target = tf.exp(target)
    target = tf.slice(target, [0, 0], [5, -1])
    values = tf.split(target, target.get_shape()[0], 0)
    out = []
    x = 0
    for val_x in values:
        y = 0
        sum = tf.zeros_like(val_x)
        for val_y in values:
            if x != y:
                if time[0][y] > time[0][x]:
                    sum = tf.add(sum, val_y)
            y += 1
        out.append(sum)
        x += 1
    result = tf.concat(out, 1) + 1
    return result

time = [[10], [5], [7], [8]]

out = cox_cummulative(target, time)

sess = tf.Session()
sess.run()