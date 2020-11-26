from constants import BATCH_SIZE, COMPETE_MODEL_EVERY, C_L2, EPOCHS, LEARNING_RATE, LOAD_DATA_EVERY, MIN_DATASET_SIZE, MOMENTUM, NO_OF_TRAININGS, SAVE_MODEL_EVERY, TRAINING_SAMPLE_SIZE
from model_path_management import *
from compare_models import compete_with_best_model
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import os
import time
from datetime import datetime
import multiprocessing as mp

def load_data():
    try:
        training_data = np.load(training_data_dir_0)
        return (training_data["state"], training_data["policy"], training_data["value"])
    except Exception:
        training_data = np.load(training_data_dir_1)
        return (training_data["state"], training_data["policy"], training_data["value"])

def loss(nn, state, policy_true, value_true, training):  # MSE + L2 loss
    [policy_pred, value_pred] = nn(state, training=training)
    weights = nn.trainable_variables
    loss_mse = tf.keras.losses.MeanSquaredError()
    loss = loss_mse(y_true=value_true, y_pred=value_pred) - \
            tf.nn.softmax_cross_entropy_with_logits(labels=policy_true, logits=policy_pred) + \
            tf.math.add_n([tf.nn.l2_loss(v) for v in weights if 'bias' not in v.name]) * C_L2
    return loss


def train(nn, state, policy_true, value_true):
    with tf.GradientTape() as g:
        loss_values = loss(nn, state, policy_true, value_true, training=True)
    return loss_values, g.gradient(loss_values, nn.trainable_variables)


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    training_data_dir_0 = training_data_dir + "_0.npz"
    training_data_dir_1 = training_data_dir + "_1.npz"
    training_results_dir = "training_results\\{}_with_L2.npz".format(MODEL_NAME)

    if os.path.exists(get_last_model_dir(model_path)):
        model = K.models.load_model(get_last_model_dir(model_path))
        print("{} loaded!".format(MODEL_NAME))
    else:
        raise FileNotFoundError("model doesn't exist")

    if os.path.exists(training_results_dir):
        training_results = np.load(training_results_dir)
        train_loss_results = training_results["loss"].tolist()
        train_mse_results = training_results["mse"].tolist()
        print("Previous training results found")
    else:
        train_loss_results = []
        train_mse_results = []
        np.savez(training_results_dir,
                 loss=np.array(train_loss_results, dtype=np.float32),
                 mse=np.array(train_mse_results, dtype=np.float32))
        print("No previous training results. Creating new.")

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    data_state = data_policy = data_value = []
    pool = mp.Pool(processes=1, maxtasksperchild=1)

    # choose data for fitting
    for TRAIN_INDEX in range(NO_OF_TRAININGS):
        if TRAIN_INDEX % LOAD_DATA_EVERY == 0:
            data_state, data_policy, data_value = load_data()
            print(f"There are {len(data_state)} data for training")
            assert len(data_state) == len(data_policy) == len(data_value), "Length of training data does not match"

        if len(data_state) > MIN_DATASET_SIZE:
            random_indices = np.random.choice(len(data_state), TRAINING_SAMPLE_SIZE, replace=False)
            selected_state = data_state[random_indices]
            selected_policy = data_policy[random_indices]
            selected_value = data_value[random_indices]

            training_dataset = tf.data.Dataset.from_tensor_slices((selected_state, selected_policy, selected_value))
            training_dataset = training_dataset.batch(BATCH_SIZE)

            # train
            for epoch in range(EPOCHS):
                start = datetime.now()
                epoch_loss_avg = tf.keras.metrics.Mean()

                # Training loop - using batches of BATCH_SIZE
                for state, policy, value in training_dataset:
                    # Optimize the model
                    loss_value, grads = train(model, state, policy, value)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    # Track progress
                    epoch_loss_avg.update_state(loss_value)  # Add current batch loss

                # End epoch
                train_loss_results.append(epoch_loss_avg.result().numpy())
                print(f"Train: {TRAIN_INDEX}, Epoch: {epoch}, Loss: {epoch_loss_avg.result():.6f}, \
                    Time: {datetime.now() - start}")
   

            if TRAIN_INDEX % SAVE_MODEL_EVERY == 0:
                try:
                    model.save(get_last_model_dir(model_path), save_format="tf")
                except Exception:
                    time.sleep(5)
                    model.save(get_last_model_dir(model_path), save_format="tf")
                print("Model saved")
                np.savez(training_results_dir,
                         loss=np.array(train_loss_results, dtype=np.float32))
                print("Training results saved")
            
            if TRAIN_INDEX % COMPETE_MODEL_EVERY == 0:
                pool.apply_async(func=compete_with_best_model, args=())
        else:
            print("Not enough data for training. There is only {} data.".format(len(data_state)))
            time.sleep(5)
            data_state, data_policy, data_value = load_data()
            print(f"There are {len(data_state)} data for training")

    print("completed!")
