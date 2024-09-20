import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove WARNING Messages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# parameter class fis parameters
class fis_parameters():
    def __init__(self, n_input: int = 6, n_memb: int = 3, batch_size: int = 16, n_epochs: int = 25, 
                 memb_func: str = 'gaussian', optimizer: str = 'adam', loss: str = 'binary_crossentropy'):
        self.n_input = n_input  
        self.n_memb = n_memb  
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.memb_func = memb_func  
        self.optimizer = optimizer  
        self.loss = loss  


# Main Class ANFIS
class ANFIS:
    def __init__(self, n_input: int, n_memb: int, batch_size: int = 16, memb_func: str = 'gaussian', name: str = 'MyAnfis'):
        self.n = n_input
        self.m = n_memb
        self.batch_size = batch_size
        self.memb_func = memb_func
        input_ = keras.layers.Input(
            shape=(n_input,), name='inputLayer', batch_size=self.batch_size)
        L1 = FuzzyLayer(n_input, n_memb, memb_func, name='fuzzyLayer')(input_)
        L2 = RuleLayer(n_input, n_memb, name='ruleLayer')(L1)
        L3 = NormLayer(name='normLayer')(L2)
        L4 = DefuzzLayer(n_input, n_memb, name='defuzzLayer')(L3, input_)
        L5 = SummationLayer(name='sumLayer')(L4)
        self.model = keras.Model(inputs=[input_], outputs=[L5], name=name)
        self.update_weights()

    def __call__(self, X):
        return self.model.predict(X, batch_size=self.batch_size)

    def update_weights(self):
        # premise parameters (mu&sigma for gaussian // a/b/c for bell-shaped)
        if self.memb_func == 'gaussian':
            self.mus, self.sigmas = self.model.get_layer(
                'fuzzyLayer').get_weights()
        elif self.memb_func == 'gbellmf':
            self.a, self.b, self.c = self.model.get_layer(
                'fuzzyLayer').get_weights()
        # consequence parameters
        self.bias, self.weights = self.model.get_layer(
            'defuzzLayer').get_weights()

    def fit(self, X, y, **kwargs):
        # save initial weights in the anfis class
        self.init_weights = self.model.get_layer('fuzzyLayer').get_weights()

        # fit model & update weights in the anfis class
        history = self.model.fit(X, y, **kwargs)
        self.update_weights()

        # clear the graphs
        tf.keras.backend.clear_session()

        return history

    def get_memberships(self, Xs):
        intermediate_layer_model = keras.Model(inputs=self.model.input,
                                               outputs=self.model.get_layer('normLayer').output)

        intermediate_L2_output = intermediate_layer_model.predict(Xs)

        return intermediate_L2_output


# Custom weight initializer
def equally_spaced_initializer(shape, minval=-1.5, maxval=1.5, dtype=tf.float32):
    """
    Custom weight initializer:
        euqlly spaced weights along an operating range of [minval, maxval].
    """
    linspace = tf.reshape(tf.linspace(minval, maxval, shape[0]),
                          (-1, 1))
    return tf.Variable(tf.tile(linspace, (1, shape[1])))


# Layer 1
class FuzzyLayer(keras.layers.Layer):
    def __init__(self, n_input, n_memb, memb_func='gaussian', **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.n = n_input
        self.m = n_memb
        self.memb_func = memb_func

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]

        if self.memb_func == 'gbellmf':
            self.a = self.add_weight(name='a',
                                     shape=(self.m, self.n),
                                     initializer=keras.initializers.RandomUniform(
                                         minval=.7, maxval=1.3, seed=1),
                                     trainable=True)
            self.b = self.add_weight(name='b',
                                     shape=(self.m, self.n),
                                     initializer=keras.initializers.RandomUniform(
                                         minval=.7, maxval=1.3, seed=1),
                                     trainable=True)
            self.c = self.add_weight(name='c',
                                     shape=(self.m, self.n),
                                     initializer=equally_spaced_initializer,
                                     trainable=True)

        elif self.memb_func == 'gaussian':
            self.mu = self.add_weight(name='mu',
                                      shape=(self.m, self.n),
                                      initializer=equally_spaced_initializer,
                                      trainable=True)
            self.sigma = self.add_weight(name='sigma',
                                         shape=(self.m, self.n),
                                         initializer=keras.initializers.RandomUniform(
                                             minval=.7, maxval=1.3, seed=1),
                                         trainable=True)

        elif self.memb_func == 'sigmoid':
            self.gamma = self.add_weight(name='gamma',
                                         shape=(self.m, self.n),
                                         initializer=equally_spaced_initializer,  
                                         trainable=True)

            self.c = self.add_weight(name='c',
                                     shape=(self.m, self.n),
                                     initializer=equally_spaced_initializer,  
                                     trainable=True)

        super(FuzzyLayer, self).build(batch_input_shape)

    def call(self, x_inputs):
        if self.memb_func == 'gbellmf':
            L1_output = 1 / (1 +
                             tf.math.pow(
                                 tf.square(tf.subtract(
                                     tf.reshape(
                                         tf.tile(x_inputs, (1, self.m)), (-1, self.m, self.n)), self.c
                                 ) / self.a), self.b)
                             )
        elif self.memb_func == 'gaussian':
            L1_output = tf.exp(-1 *
                               tf.square(tf.subtract(
                                   tf.reshape(
                                       tf.tile(x_inputs, (1, self.m)), (-1, self.m, self.n)), self.mu
                               )) / tf.square(self.sigma))

        elif self.memb_func == 'sigmoid':
            L1_output = tf.math.divide(1,
                                       tf.math.exp(-self.gamma *
                                                   tf.subtract(
                                                       tf.reshape(
                                                           tf.tile(x_inputs, (1, self.m)), (-1, self.m, self.n)), self.c)
                                                   )
                                       )
        return L1_output


# Layer 2
class RuleLayer(keras.layers.Layer):
    def __init__(self, n_input, n_memb, **kwargs):
        super(RuleLayer, self).__init__(**kwargs)
        self.n = n_input  
        self.m = n_memb  
        self.batch_size = None

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]
        super(RuleLayer, self).build(batch_input_shape)

    def call(self, input_):
        batch_size = tf.shape(input_)[0]  
        # Initialize L2_output with the first input feature reshaped appropriately
        L2_output = tf.reshape(input_[:, :, 0], [batch_size] + [-1] + [1] * (self.n - 1))

        # Iterate over the remaining input features and multiply them in sequence
        for i in range(1, self.n):
            L2_output *= tf.reshape(input_[:, :, i], [batch_size] + [1] * i + [-1] + [1] * (self.n - i - 1))

        # Flatten the resulting tensor
        return tf.reshape(L2_output, [batch_size, -1])


# Layer 3
class NormLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, w):
        w_sum = tf.reshape(tf.reduce_sum(w, axis=1), (-1, 1))
        w_norm = w / w_sum
        return w_norm


# Layer 4
class DefuzzLayer(keras.layers.Layer):
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = n_input
        self.m = n_memb

        self.CP_bias = self.add_weight(name='Consequence_bias',
                                       shape=(1, self.m ** self.n),
                                       initializer=keras.initializers.RandomUniform(
                                           minval=-2, maxval=2),
                                       trainable=True)
        self.CP_weight = self.add_weight(name='Consequence_weight',
                                         shape=(self.n, self.m ** self.n),
                                         initializer=keras.initializers.RandomUniform(
                                             minval=-2, maxval=2),
                                         trainable=True)

    def call(self, w_norm, input_):

        L4_L2_output = tf.multiply(w_norm,
                                   tf.matmul(input_, self.CP_weight) + self.CP_bias)
        return L4_L2_output  # Defuzzyfied Layer


# Layer 5
class SummationLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]
        super(SummationLayer, self).build(batch_input_shape)

    def call(self, input_):
        L5_L2_output = tf.reduce_sum(input_, axis=1)
        L5_L2_output = tf.reshape(L5_L2_output, (-1, 1))
        return L5_L2_output

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained ANFIS model with custom objects
model = load_model('ANFIS_tuned_model.h5', custom_objects={
    'FuzzyLayer': FuzzyLayer,
    'RuleLayer': RuleLayer,
    'NormLayer': NormLayer,
    'DefuzzLayer': DefuzzLayer,
    'SummationLayer': SummationLayer
})

# Streamlit app title
st.title("Automobile Insurance Fraud Prediction")

# Sidebar for user input section
st.sidebar.header("Input Features")

# Create input form for each feature
def user_input_features():
    make = st.sidebar.number_input('Make', min_value=0, max_value=100)  # Adjust based on the actual range
    accident_area = st.sidebar.number_input('Accident Area', min_value=0, max_value=1)
    month_claimed = st.sidebar.number_input('Month Claimed', min_value=0, max_value=12)
    sex = st.sidebar.number_input('Sex', min_value=0, max_value=1)
    marital_status = st.sidebar.number_input('Marital Status', min_value=0, max_value=1)
    fault = st.sidebar.number_input('Fault', min_value=0, max_value=1)
    vehicle_category = st.sidebar.number_input('Vehicle Category', min_value=0, max_value=10)
    vehicle_price = st.sidebar.number_input('Vehicle Price', min_value=0, max_value=10)  # Based on binary transformation
    deductible = st.sidebar.number_input('Deductible', min_value=0, max_value=1000)
    driver_rating = st.sidebar.number_input('Driver Rating', min_value=0, max_value=5)
    days_policy_accident = st.sidebar.number_input('Days Policy Accident', min_value=0, max_value=500)
    days_policy_claim = st.sidebar.number_input('Days Policy Claim', min_value=0, max_value=500)
    past_number_of_claims = st.sidebar.number_input('Past Number of Claims', min_value=0, max_value=10)
    age_of_vehicle = st.sidebar.number_input('Age of Vehicle', min_value=0, max_value=10)
    police_report_filed = st.sidebar.number_input('Police Report Filed', min_value=0, max_value=1)
    witness_present = st.sidebar.number_input('Witness Present', min_value=0, max_value=1)
    agent_type = st.sidebar.number_input('Agent Type', min_value=0, max_value=1)
    number_of_suppliments = st.sidebar.number_input('Number of Supplements', min_value=0, max_value=5)
    address_change_claim = st.sidebar.number_input('Address Change Claim', min_value=0, max_value=3)
    number_of_cars = st.sidebar.number_input('Number of Cars', min_value=0, max_value=10)
    year = st.sidebar.number_input('Year', min_value=1990, max_value=2024)
    base_policy = st.sidebar.number_input('Base Policy', min_value=0, max_value=2)
    age_group = st.sidebar.number_input('Age Group', min_value=0, max_value=10)

    # Package inputs into a DataFrame or numpy array as expected by the model
    features = [make, accident_area, month_claimed, sex, marital_status, fault, vehicle_category,
                vehicle_price, deductible, driver_rating, days_policy_accident, days_policy_claim,
                past_number_of_claims, age_of_vehicle, police_report_filed, witness_present,
                agent_type, number_of_suppliments, address_change_claim, number_of_cars, year,
                base_policy, age_group]
    
    return np.array(features).reshape(1, -1)

# Get user inputs
input_features = user_input_features()

# Predict using the loaded model
if st.button('Predict'):
    prediction = model.predict(input_features)
    # Assuming the prediction output is binary (0 or 1 for fraud classification)
    st.write(f'Prediction: {"Fraud" if prediction[0][0] == 1 else "No Fraud"}')
    
    # Optionally, display the raw input data
    st.subheader("Input Data")
    st.write(pd.DataFrame(input_features, columns=['Make', 'Accident Area', 'Month Claimed', 'Sex', 'Marital Status', 
                                                   'Fault', 'Vehicle Category', 'Vehicle Price', 'Deductible', 
                                                   'Driver Rating', 'Days Policy Accident', 'Days Policy Claim', 
                                                   'Past Number of Claims', 'Age of Vehicle', 'Police Report Filed', 
                                                   'Witness Present', 'Agent Type', 'Number of Supplements', 
                                                   'Address Change Claim', 'Number of Cars', 'Year', 'Base Policy', 
                                                   'Age Group']))
