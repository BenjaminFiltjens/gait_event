# Seed environment to generate reproducible results
# Seed value
seed_value = 41

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from FileSelectGui3 import getDirectoryPath
import argparse
import pickle
from hyperopt import hp, tpe, fmin, Trials, space_eval, STATUS_OK
import sklearn.model_selection as ms
from sklearn.utils.class_weight import compute_class_weight

os.environ['PYTHONHASHSED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.recurrent import *
import tf_models
from sklearn.preprocessing import MinMaxScaler
from detect_peaks import detect_peaks
from sklearn.utils import shuffle

from keras import backend as K
config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.set_random_seed(seed_value)
sess = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(sess)


def reset_seeds(reset_graph_with_backend=None):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional

    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)
    print("RANDOM SEEDS RESET")


# This is the space of hyperparameters that we will optimize for
space = {
    'num_filters': hp.choice('num_filters', [2, 4, 8, 16, 32]),
    'num_dil': hp.choice('num_dil', [[1], [1,1]]),
}
LSTM = False


# Load CSV file with kinematics and markers. Interpret last columns as outcome
# nseqlen defines the length of the input window
def load_seq(R, nstart, nseqlen, input_dim, output_dim):
    X = R[nstart:(nstart + nseqlen), 0:36]
    Y = R[nstart:(nstart + nseqlen), 36:(36 + output_dim)]
    return X, Y.astype(int)[:, 0:output_dim]


def load_file(filename, input_dim, output_dim, nseqlen=128):
    try:
        R = np.loadtxt(filename, delimiter=',')
    except:
        return None

    positives = np.where(R[:, input_dim] > 0.5)
    if len(positives[0]) == 0:
        return None

    nstart = 0
    nend = len(R)

    nstart = nstart

    nobs = int(((nend - nstart) / nseqlen))

    # Create labels of the 15 patients for the leave one subject out split (hardcoded based on filenames)
    if "0" in filename:
        id = 0
    if "1" in filename:
        id = 1
    if "2" in filename:
        id = 2
    if "3" in filename:
        id = 3
    if "4" in filename:
        id = 4
    if "5" in filename:
        id = 5
    if "6" in filename:
        id = 6
    if "7" in filename:
        id = 7
    if "8" in filename:
        id = 8
    if "9" in filename:
        id = 9
    if "10" in filename:
        id = 10
    if "11" in filename:
        id = 11
    if "12" in filename:
        id = 12
    if "13" in filename:
        id = 13
    if "14" in filename:
        id = 14

    if R.shape[0] < (nstart + nseqlen):
        return None

    inputs_file = np.zeros((nobs, nseqlen, input_dim))
    outputs_file = np.zeros((nobs, nseqlen, output_dim))
    ids = np.zeros((nobs, output_dim))
    for j in range(nobs):
        data = load_seq(R, nstart, nseqlen, input_dim, output_dim)
        X, Y = data
        sag_1 = X[:,0]
        sag_2 = X[:,3]
        sag_3 = X[:,6]
        sag_4 = X[:,9]
        sag_5 = X[:,12]
        sag_6 = X[:,15]
        sag_7 = X[:,18]
        sag_8 = X[:,21]
        sag_9 = X[:,24]
        sag_10 = X[:,27]
        sag_11 = X[:,30]
        sag_12 = X[:,33]
        X = np.vstack((sag_1, sag_2, sag_3, sag_4, sag_5, sag_6, sag_7, sag_8, sag_9, sag_10, sag_11, sag_12))
        X = np.swapaxes(X, 0, 1)
        inputs_file[j, :, :] = X
        outputs_file[j, :, :] = Y
        ids[j, :] = id
        nstart += nseqlen

    return inputs_file, outputs_file, ids


## Load all files from a given directory
def load_data(fdir, input_dim, output_dim, nseqlen, nsamples=100000):
    files = os.listdir(fdir)

    # Merge inputs from different files together
    inputs = np.empty((0, nseqlen, input_dim))
    outputs = np.empty((0, nseqlen, output_dim))
    ids = np.empty((0, output_dim))

    n = 0
    for i, filename in enumerate(files):
        fname = "%s/%s" % (fdir, filename)

        data = load_file(fname, input_dim, output_dim, nseqlen)
        if not data:
            continue
        X, Y, id = data

        inputs = np.concatenate([inputs, X])
        outputs = np.concatenate([outputs, Y])

        ids = np.concatenate([ids, id])
        n = n + 1

        if n >= nsamples:
            break

    return inputs, outputs, ids


# Learning rate decay (from https://github.com/kidzik/event-detector-train)
def sigmoidal_decay(e, start=0, end=100, lr_start=1e-3, lr_end=1e-5):
    if e < start:
        return lr_start

    if e > end:
        return lr_end

    middle = (start + end) / 2
    s = lambda x: 1 / (1 + np.exp(-x))

    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end


# run an experiment
def run_experiment():
    nseqlen = 128
    # load data
    trainX, trainY, ids = load_data(train_dir, input_dim, output_dim, nseqlen, nsamples=100000)
    testX, testY, ids_test = load_data(test_dir, input_dim, output_dim, nseqlen, nsamples=100000)

    # Shuffle samples
    trainX, trainY, ids = shuffle(trainX, trainY, ids)
    testX, testY, ids_test = shuffle(testX, testY, ids_test)

    # Train as a binary classifier
    to = False
    # As determined by the read.py script:
    # Left heel strike = 1
    # Right Heel strike = 2
    # Left toe off = 3
    # Right toe off = 4
    if to:
        trainY = (trainY == 3).astype(int)
        testY = (testY == 3).astype(int)
    else:
        trainY = (trainY == 1).astype(int)
        testY = (testY == 1).astype(int)

    scale = False
    if scale:
        scaler = MinMaxScaler()

        num_instances, num_time_steps, num_features = trainX.shape
        train_data = np.reshape(trainX, (-1, num_features))
        fitA = scaler.fit(train_data)
        train_data = fitA.transform(train_data)
        trainX = np.reshape(train_data, (num_instances, num_time_steps, num_features))

        num_instances, num_time_steps, num_features = testX.shape
        test_data = np.reshape(testX, (-1, num_features))
        test_data = fitA.transform(test_data)
        testX = np.reshape(test_data, (num_instances, num_time_steps, num_features))

    return trainX, trainY, ids, testX, testY, ids_test


def create_model(trainX, trainY, patient, space, verbose, final):
        # Get data structure
        n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
        n_outputs = 1
        print("trainData length: %d" % trainX.shape[0])
        print('Temporal domain: ', n_timesteps, ', Spatial domain: ', n_features, ', Output dim: ', n_outputs)

        # Get hyperparameters
        n_feature_maps = space['num_filters']
        n_nodes = [n_feature_maps, n_feature_maps * 2]
        n_kernels = 5
        n_dil = space['num_dil']
        n_layers = len(n_dil)
        epochs = 150
        drop = 0.25
        mini_batch_size = 32

        # Callback functions
        mcp_save = ModelCheckpoint("models_LSTM_ud/" + str(patient) + '_HS.h5', save_best_only=False, mode='min')

        # Not used since it requires keras adam optimizer, which results in not reproducable results
        #lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, lr_start=0.01, end=epochs))

        # Compute class weight
        weightY = trainY.flatten()
        class_weight = compute_class_weight('balanced', np.unique(weightY), weightY)
        class_weight = class_weight[0]/class_weight[1]
        print("weight: " + str(class_weight))

        # Get models
        if LSTM:
            model = tf_models.uniDirLSTM(n_nodes[0], drop, n_outputs, input_dim, n_layers, class_weight, n_timesteps, causal=True)
        else:
            model = tf_models.TCN_keras(n_nodes[0], drop, n_outputs, input_dim, n_kernels, class_weight, n_timesteps, dilations=n_dil, causal=True)

        if final:
            callbacks = [mcp_save]
        else:
            callbacks = []

        model.fit(x=trainX, y=trainY, epochs=epochs, batch_size=mini_batch_size, shuffle=True, verbose=verbose, callbacks=callbacks)

        return model


# Adjusted from kidzinski et al.
def peak_cmp(annotated, predicted):
    score = 0
    predicted = [k for k in predicted if (k >= 10 and k < 128 - 10)]
    annotated = [k for k in annotated if (k >= 10 and k < 128 - 10)]

    if len(predicted) == 1:
        tresh = 15
    else:
        tresh = 30

    if (len(predicted) != len(annotated)):
        score -= 1
    if (len(predicted) == len(annotated)):
        score += 1
        dist = 0
        for a in range(len(annotated)):
            dist += (np.abs(predicted[a] - annotated[a]))
            if (dist > tresh):
                score -= 1
    return score


def results(model, valX, valY):
    scores = model.evaluate(x=valX, y=valY)
    loss = scores[0]
    bin_acc = scores[1]

    y_pred = model.predict(valX)

    sdist = []
    for ntrial in range(len(y_pred)):
        likelihood = y_pred[ntrial, :, 0]
        true = valY[ntrial, :, 0]

        peakind = detect_peaks(likelihood, mph=0.5, mpd=15)
        true = np.where(true > 0.5)
        sdist.append(peak_cmp(true[0], peakind))

    score = np.mean(sdist)
    print('bin_acc: %f' % bin_acc)
    print('score: %f' % score)

    return y_pred, valY, score


def optimize(space):
    reset_seeds(K)
    trainX, trainY, cv_labels, testX, testY, cv_labels_test = run_experiment()

    cv_labels = cv_labels.flatten()
    splitTrain = trainX[:,2]
    splitTrainY = trainY[:,1]
    group_kfold = ms.LeaveOneGroupOut()
    group_kfold.get_n_splits(splitTrain, splitTrainY, groups=cv_labels)
    cvscores = []
    for train_idx, test_idx in group_kfold.split(splitTrain, splitTrainY, cv_labels):
        trainX2 = trainX[train_idx]
        trainY2 = trainY[train_idx]
        valX2 = trainX[test_idx]
        valY2 = trainY[test_idx]
        model = create_model(trainX2, trainY2, patient=cv_labels[test_idx][0], space=space, verbose=0, final=False)
        y_pred, y_true, result = results(model, valX2, valY2)
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        print("Validation subject: " + str(cv_labels[test_idx][0]))
        np.savetxt('models_LSTM_ud/' + str(cv_labels[test_idx][0]) + '_HS_pred', y_pred)
        np.savetxt('models_LSTM_ud/' + str(cv_labels[test_idx][0]) + '_HS_true', y_true)
        print("Loss: " + str(result))
        cvscores.append(result)
        del model
        reset_seeds(K)

    score_avg = np.mean(cvscores)
    print("Average loss: " + str(score_avg))
    print('PARAM: ' + str(space))

    return {'loss': (1-score_avg), 'status': STATUS_OK}


def optimize_final(space):
    reset_seeds(K)
    trainX, trainY, cv_labels, testX, testY, cv_labels_test = run_experiment()
    test_idx = '7.0'

    model = create_model(trainX, trainY, patient=test_idx, space=space, verbose=2, final=True)
    y_pred, y_true, result = results(model, testX, testY)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    print("Validation subject: " + str(cv_labels[test_idx][0]))
    print("Loss: " + str(result))
    return y_pred, y_true, result, model


if __name__ == "__main__":

    # Define global variables
    global save
    global first
    global filename

    # Get command line arguments
    parser = argparse.ArgumentParser(description='Label c3d files')
    parser.add_argument('--save', dest='save', action='store_true',
                        default=False,
                        help='Save the resulting plots (default: do not save)')
    parser.add_argument('--directory', dest='directory', action='store_true',
                        default=False,
                        help='Process all .c3d files in a directory (default: process a single file)')

    args = parser.parse_args()
    save = args.save
    directory = args.directory

    if save:
        prefix_results = getDirectoryPath('Select results directory')

    if not directory:
        # Set parameters (input dim = 36 + label patient)
        # Only sagittal plane kinematics are used (12 dim)
        input_dim = 12
        output_dim = 1

        train_dir = getDirectoryPath('Train directory')
        test_dir = getDirectoryPath('Test directory')

        trials = Trials()

        # Hyperparameter turning
        best = fmin(fn=optimize, space=space, algo=tpe.suggest, trials=trials, max_evals=10)
        #
        # The trials database now contains 10 entries, it can be saved/reloaded with pickle or another method
        pickle.dump(trials, open("HS_Left.p", "wb"))

        # Print best parameters
        best_params = space_eval(space, best)
        print("BEST PARAMETERS: " + str(best_params))

        # Set params (if loading parameters from the pickle)
        #pickle_in = open("HS_right.p", "rb")
        #best_params = pickle.load(pickle_in)

        # After tuning, train the best model on the left out test subject
        optimize_final(space=best_params)



