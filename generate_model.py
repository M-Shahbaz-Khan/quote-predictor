import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from datetime import datetime
from packaging import version
import random
import tensorboard

np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 60)

def remove_fields(df):
    df.columns = list(map(lambda x: str.replace(x, 'fields.', ''), df.columns.values))
    return df

############################# Load Data

quotes_csv = pd.read_csv('C:\\Users\\shahb\\Documents\\Glow\\lead-prospect-data-discovery\\quote_predictor\\quotes.csv')

data = pd.concat([pd.json_normalize(quotes_csv.quote.apply(lambda x: eval(x.replace('false', 'False').replace('true', 'True').replace('null', 'None')))),
                    pd.json_normalize(quotes_csv.application.apply(lambda x: eval(x.replace('false', 'False').replace('true', 'True').replace('null', 'None'))))], axis=1)

data = pd.concat([data,
                    pd.json_normalize(quotes_csv.lead.apply(lambda x: eval(x.replace('false', 'False').replace('true', 'True').replace('null', 'None'))))], axis=1)

############################# Clean/Filter Data

cleaned_data_list = []
for idx, row in data.iterrows():
    carrier = row.carrier
    xmod = row['currentx_mod']

    geographicRegion = random.randint(1,19) # replace with postal code mapping

    for class_rating in row['workers_compensation.classRatings']:
        class_code = class_rating['classCode']
        exposure = class_rating['exposure']
        net_rate = class_rating['adjustedRate'] # replace with calculated net rate

        cleaned_data_list.append({
            'carrier' : carrier,
            'xmod' : xmod,
            'class_code' : class_code,
            'exposure' : exposure,
            'net_rate' : net_rate,
            'wcirbGeographicRegion' : geographicRegion
        })

dataset = pd.DataFrame(cleaned_data_list)

dataset.loc[:, 'xmod'] = dataset.xmod.fillna(0.0).astype(np.int32)
dataset.loc[:, 'exposure'] = dataset.exposure.fillna(0.0).astype(np.int32)

allowed_codes = set(['9079', '9079B', '8017', '8810', '8078', '9079A'])

dataset.loc[:, 'net_rate'] = dataset.net_rate.fillna(2.25)
dataset = dataset[dataset.class_code.isin(allowed_codes)].copy()
dataset.reset_index(inplace=True, drop=True)

dataset_dummies = pd.get_dummies(dataset, columns=['class_code', 'carrier', 'wcirbGeographicRegion'], prefix='', prefix_sep='')

############################# Train/Test Split

train_dataset = dataset_dummies.sample(frac=0.8, random_state=0)
test_dataset = dataset_dummies.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('net_rate')
test_labels = test_features.pop('net_rate')

############################# Build & Train Model

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

linear_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(33)),
    normalizer,
    tf.keras.layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

logdir=".\\logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=10,
    verbose=0,
    validation_split = 0.2,
    callbacks=[tensorboard_callback])

linear_model.save('quote_predictor_model')

############################# Check Results

test_results = {}

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(test_features, test_labels, verbose=1)
tf.keras.utils.plot_model(linear_model, to_file='.\\graph.png', show_shapes=True)

linear_model.predict(train_features[:1])
train_features.to_csv('features.csv')