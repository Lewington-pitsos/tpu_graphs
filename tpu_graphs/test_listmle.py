import tensorflow as tf
import tensorflow_ranking as tfr

mle = tfr.keras.losses.ListMLELoss()


ground_truth = tf.constant([[1.0, 0.0]])
preds = tf.constant([[0.1, 2.6]])
score = mle(ground_truth, preds)
print(score.numpy())
