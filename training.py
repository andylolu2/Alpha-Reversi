from model_path_management import *
from find_best_model import compete_with_best_model
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import os
import time
from datetime import datetime
import multiprocessing as mp


if __name__ == '__main__':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    NO_OF_TRAININGS = 10_000
    SAVE_EVERY = 50
    LOAD_DATA_EVERY = 25
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    C_L2 = 1e-4
    EPOCHS = 1

    training_data_dir_0 = training_data_dir + "_0.npz"
    training_data_dir_1 = training_data_dir + "_1.npz"
    training_results_dir = "training_results\\{}_with_L2.npz".format(MODEL_NAME)

    if os.path.exists(get_last_model_dir(model_path)):
        model = K.models.load_model(get_last_model_dir(model_path))
        print("{} loaded!".format(MODEL_NAME))
    else:
        print("model doesn't exist")
        exit()

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    loss_mse = tf.keras.losses.MeanSquaredError()


    def loss(nn, x_, y_true, training):  # MSE + L2 loss
        y_pred = nn(x_, training=training)
        weights = nn.trainable_variables
        loss = loss_mse(y_true=y_true, y_pred=y_pred) + \
               tf.math.add_n([tf.nn.l2_loss(v) for v in weights if 'bias' not in v.name]) * C_L2
        return loss, y_pred


    def train(nn, x, y):
        with tf.GradientTape() as g:
            loss_values, y_pred = loss(nn, x, y, training=True)
        return loss_values, g.gradient(loss_values, nn.trainable_variables), y_pred


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

    training_data = None
    pool = mp.Pool(processes=1, maxtasksperchild=1)
    # choose data for fitting
    for TRAIN_INDEX in range(NO_OF_TRAININGS):
        if TRAIN_INDEX % LOAD_DATA_EVERY == 0:
            try:
                training_data = np.load(training_data_dir_0)
                x_train = training_data["x_train"]
                y_train = training_data["y_train"]
                print("There are {} data for training".format(len(x_train)))
            except:
                training_data = np.load(training_data_dir_1)
                x_train = training_data["x_train"]
                y_train = training_data["y_train"]
                print("There are {} data for training".format(len(x_train)))
            assert len(x_train) == len(y_train)
        if len(x_train) > 2048 * 4:
            random_indices = np.random.choice(len(x_train), 4096, replace=False)
            selected_x = x_train[random_indices]
            selected_y = y_train[random_indices]

            training_dataset = tf.data.Dataset.from_tensor_slices((selected_x, selected_y))
            training_dataset = training_dataset.batch(BATCH_SIZE)

            # train
            for epoch in range(EPOCHS):
                start = datetime.now()
                epoch_loss_avg = tf.keras.metrics.Mean()
                epoch_mse = tf.keras.metrics.MeanSquaredError()

                # Training loop - using batches of 32
                for x, y in training_dataset:
                    # Optimize the model
                    loss_value, grads, y_pred = train(model, x, y)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    # Track progress
                    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                    epoch_mse.update_state(y, y_pred)

                # End epoch
                train_loss_results.append(epoch_loss_avg.result().numpy())
                train_mse_results.append(epoch_mse.result().numpy())
                time_taken = datetime.now() - start
                if epoch % 1 == 0:
                    print("Train: {}, Epoch: {}, Loss: {:.6f}, MSE: {:.6f}, Time: {}".format(TRAIN_INDEX,
                                                                                             epoch,
                                                                                             epoch_loss_avg.result(),
                                                                                             epoch_mse.result(),
                                                                                             time_taken))

            if TRAIN_INDEX % SAVE_EVERY == 0:
                try:
                    model.save(get_next_model_dir(model_path), save_format="tf")
                except:
                    time.sleep(5)
                    model.save(get_next_model_dir(model_path), save_format="tf")
                print("Model saved")
                np.savez(training_results_dir,
                         loss=np.array(train_loss_results, dtype=np.float32),
                         mse=np.array(train_mse_results, dtype=np.float32))
                print("Training results saved")
                pool.apply_async(func=compete_with_best_model, args=())
        else:
            print("Not enough data for training. There is only {} data.".format(len(x_train)))
            time.sleep(5)
            try:
                training_data = np.load(training_data_dir_0)
                x_train = training_data["x_train"]
                y_train = training_data["y_train"]
                print("There are {} data for training".format(len(x_train)))
            except:
                training_data = np.load(training_data_dir_1)
                x_train = training_data["x_train"]
                y_train = training_data["y_train"]
                print("There are {} data for training".format(len(x_train)))

    print("completed!")
