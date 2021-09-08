import tensorflow as tf
import tensorflow.keras as keras
from einops import repeat

class TemporalAttentionModule(keras.Model):
    def __init__(self, featureNum, reduction_ratio=2):
        """This is the structure of temporal attention module

        Args:
            featureNum (int): It means the number of hidden units of RNNs.
            reduction_ratio (int): The reduction ratio of number of hidden units. Defaults to 2.
        """
        super(TemporalAttentionModule, self).__init__()
        
        self.scoreLayer = keras.Sequential([
            keras.layers.Dense(featureNum//reduction_ratio, activation=None),
            keras.layers.LayerNormalization(),
            keras.layers.Activation(tf.nn.relu),
            keras.layers.Dense(1, activation=None),
            keras.layers.Flatten()            
        ])

        self.softmax = keras.layers.Softmax(axis=-1)

    def call(self, x):
        """call function of TAM

        Args:
            x (array): The input data after RNNs, of which its dimension should be (batch x T x H).

        Returns:
            x (array): The output data after aggregating, of which its dimension would be same as inputs.
            score (array): attention weights of TAM
        """
        score = self.scoreLayer(x)

        score = self.softmax(score)
        score = repeat(score, 'bs t -> bs t a', a=1)
        x = tf.multiply(x, tf.broadcast_to(score, tf.shape(x)))
        x = tf.reduce_sum(x, axis=1, keepdims=True)

        return x, score

class lstm_decoder(keras.Model):
    def __init__(self, tapsize, attn=True):
        """This is the structure of LSTM-based neural decoder

        Args:
            tapsize (int): It means number of timesteps
            attn (bool, optional): Whether use TAM or not. Defaults to True.
        """
        super(lstm_decoder, self).__init__()

        # params
        self.tapsize = tapsize
        self.attn = attn

        # layers
        self.emb = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True), merge_mode='concat'),
            keras.layers.LayerNormalization(),
            keras.layers.LSTM(256, return_sequences=True)
        ])   

        if attn:
            self.tmpAttn = TemporalAttentionModule(featureNum=256, reduction_ratio=2)

        self.decoder = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            if self.attn:
                y_pred, scale = self(x, training=True)  # Forward pass
            else:
                y_pred = self(x, training=True)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        """call function of LSTM

        Args:
            x (array): Firing rate, of which its dimension should be (batch x T x channel).

        Returns:
            x (array): Kinematic state, and its dimension should be (batch x 1). 
                       That means you only can predict one dimension of kinematics once. 
        """
        x = x[:, -self.tapsize:, :]
        x = self.emb(x)   
        
        if self.attn:
            x, scale = self.tmpAttn(x)
            x = self.decoder(x)

            return x, scale
        else:
            x = x[:, -1, :]
            x = self.decoder(x)
        
            return x