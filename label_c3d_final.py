import pandas as pd
from FileSelectGui import getDirectoryPath, getFilePath
import argparse
from matplotlib import pyplot as plt

# Seed value
seed_value = 41

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
from detect_peaks import detect_peaks
import btk
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
import sys

np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
from keras.models import load_model
from scipy.signal import find_peaks

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def weighted_binary_crossentropy(y_true, y_pred):
    a1 = K.mean(np.multiply(K.binary_crossentropy(y_pred, y_true),(y_true + 0.01)), axis=-1)
    return a1


marker = False

modelFO = load_model("final_models/models_LSTM_ud/14.0_TO.h5", custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy}, compile = False)
modelHS = load_model("final_models/models_LSTM_ud/14.0_HS.h5", custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy}, compile = False)


def derivative(traj, nframes):
    traj_der = traj[1:nframes, :] - traj[0:(nframes - 1), :]
    shape = [[0, 0, 0] * (traj.shape[1] / 3)]
    return np.append(traj_der, shape, axis=0)


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def extract_kinematics(leg, filename_in, sacr=False, lstm=True, turn=False):
    # Open c3d and read data
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename_in)
    reader.Update()
    acq = reader.GetOutput()
    nframes = acq.GetPointFrameNumber()
    first_frame = acq.GetFirstFrame()

    end = acq.GetLastFrame()

    # Check if there is a FOG
    for event in btk.Iterate(acq.GetEvents()):
        if event.GetLabel() == 'FOG':
            end = event.GetFrame()
            nframes = end - first_frame

    metadata = acq.GetMetaData()

    # We extract only kinematics
    kinematics = ["HipAngles", "KneeAngles", "AnkleAngles"]
    markers = ["TOE", "KNE", "HEE"]

    # Check if there are any kinematics in the file
    brk = True
    for point in btk.Iterate(acq.GetPoints()):
        if point.GetLabel() == "L" + kinematics[0]:
            brk = False
            break

    if brk:
        print("No kinematics in:" + str(filename_in))
        return

    # Add events as output
    outputs = np.array([[0] * nframes]).T
    for event in btk.Iterate(acq.GetEvents()):
        if first_frame < event.GetFrame() < end:
            if event.GetLabel() == "Foot Strike":
                if event.GetContext() == 'Left':
                    outputs[event.GetFrame() - first_frame, 0] = 1
                elif event.GetContext() == 'Right':
                    outputs[event.GetFrame() - first_frame, 0] = 2
            elif event.GetLabel() == "Foot Off":
                if event.GetContext() == 'Left':
                    outputs[event.GetFrame() - first_frame, 0] = 3
                elif event.GetContext() == 'Right':
                    outputs[event.GetFrame() - first_frame, 0] = 4

    if (np.sum(outputs) == 0):
        print("No events in:" + str(filename_in))
        return

    positives = np.where(outputs > 0.5)
    if len(positives[0]) == 0:
        return None

    first_event = positives[0][0]

    # Combine kinematics into one big array
    opposite = {'L': 'R', 'R': 'L'}
    angles = [None] * (len(kinematics) * 2)
    for i, v in enumerate(kinematics):
        point = acq.GetPoint(leg + v)
        angles[i] = point.GetValues()
        angles[i] = angles[i][:nframes]
        point = acq.GetPoint(opposite[leg] + v)
        angles[len(kinematics) + i] = point.GetValues()
        angles[len(kinematics) + i] = angles[len(kinematics) + i][:nframes]

    print(filename_in)

    # Get the pelvis (if no SACR marker)
    if sacr:
        SACR_X = acq.GetPoint("SACR").GetValues()[:, 0]
    else:
        LPSI_X = acq.GetPoint("LPSI").GetValues()[:, 0]
        RPSI_X = acq.GetPoint("RPSI").GetValues()[:, 0]
        midPSI_X = (LPSI_X + RPSI_X) / 2
        SACR_X = midPSI_X

    pos = [None] * (len(markers) * 2)

    pos_sacr_X = [None] * (len(markers) * 2)
    pos_sacr_Y = [None] * (len(markers) * 2)
    pos_sacr_Z = [None] * (len(markers) * 2)
    pos_sacr = [None] * (len(markers) * 2)
    for j, w in enumerate(markers):
        point = acq.GetPoint(leg + w)
        pos[j] = point.GetValues()
        pos_sacr_X[j] = point.GetValues()[:,0] - SACR_X
        pos_sacr_Y[j] = point.GetValues()[:,1]
        pos_sacr_Z[j] = point.GetValues()[:,2]
        pos_sacr[j] = np.column_stack((pos_sacr_X[j], pos_sacr_Y[j], pos_sacr_Z[j]))
        pos_sacr[j] = pos_sacr[j][:nframes]

        point = acq.GetPoint(opposite[leg] + w)
        pos[len(markers) + j] = point.GetValues()
        pos_sacr_X[len(markers) + j] = point.GetValues()[:,0] - SACR_X
        pos_sacr_Y[len(markers) + j] = point.GetValues()[:,1]
        pos_sacr_Z[len(markers) + j] = point.GetValues()[:,2]
        pos_sacr[len(markers) + j] = np.column_stack((pos_sacr_X[len(markers)+ j], pos_sacr_Y[len(markers) + j], pos_sacr_Z[len(markers) + j]))
        pos_sacr[len(markers) + j] = pos_sacr[len(markers) + j][:nframes]


    # Low pass filter at 7Hz
    angles_lp = butter_lowpass_filter(np.hstack(angles), cutoff=7, fs=100)
    markers_lp = butter_lowpass_filter(np.hstack(pos_sacr), cutoff=7, fs=100)

    # Get derivatives
    angles_lp_vel = derivative(angles_lp, nframes)
    markers_lp_vel = derivative(markers_lp, nframes)

    angles = np.hstack((angles_lp, angles_lp_vel))
    markers = np.hstack((markers_lp, markers_lp_vel))

    if marker:
        curves = np.concatenate((angles, markers), axis=1)
    else:
        curves = angles

    arr = np.concatenate((curves, outputs), axis=1)

    # Remove data before and after first event minus some random int. This is for those trials that are not pre-cut.
    if sacr:
        positives = np.where(arr[:, 36] > 0.5)
        if len(positives[0]) == 0:
            return None

        first_event = positives[0][0] - random.randint(5, 15)
        last_event = positives[0][-1] + random.randint(5, 15)

        curves = curves[first_event:last_event]
        first_frame = first_event

    out = [curves, markers, angles, first_frame]
    return out


def load_seq(R, nstart, nseqlen, input_dim):
    X = R[nstart:(nstart + nseqlen), 0:input_dim]
    return X


def ml_method(inputs, model, first_frame, file):
    input_dim = 12

    X = inputs
    sag_1 = X[:, 0]
    sag_2 = X[:, 3]
    sag_3 = X[:, 6]
    sag_4 = X[:, 9]
    sag_5 = X[:, 12]
    sag_6 = X[:, 15]
    sag_7 = X[:, 18]
    sag_8 = X[:, 21]
    sag_9 = X[:, 24]
    sag_10 = X[:, 27]
    sag_11 = X[:, 30]
    sag_12 = X[:, 33]
    X = np.vstack((sag_1, sag_2, sag_3, sag_4, sag_5, sag_6, sag_7, sag_8, sag_9, sag_10, sag_11, sag_12))
    X = np.swapaxes(X, 0, 1)
    X = X.reshape((1,inputs.shape[0],input_dim))

    y_pred = model.predict(x=X)
    likelihood = y_pred[0, :, 0]
    likelihood[likelihood > 0.5] = 1
    likelihood[likelihood <= 0.5] = 0
    likelihood[0] = 0
    likelihood[-1] = 0

    peakind, properties = find_peaks(likelihood, height=0.5)
    frames_window = []
    frames = []
    if peakind.any():
        for frame in peakind:
            frames_window.append(frame)
            frame += first_frame
            frame.astype(np.int32)
            frames.append(frame)

    return frames


def process(filename_in, filename_out, sacr=True, debug=True):
    filename_in = filename_in.encode()
    filename_out = filename_out.encode()

    curves, markers, kin, first_frame = extract_kinematics('L', filename_in, sacr=sacr)

    events = {}
    events[("Foot Strike", "Left")] = ml_method(curves, modelHS, first_frame, filename_out)
    events[("Foot Off", "Left")] = ml_method(curves, modelFO, first_frame, filename_out)

    if debug:
        plt.show()

    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename_in)
    reader.Update()
    acq = reader.GetOutput()

    acq.ClearEvents()

    print("First_frame: " + str(first_frame))

    for k, v in events.items():
        for frame in v:
            fps = 100.0
            event = btk.btkEvent()
            event.SetLabel(k[0])
            event.SetContext(k[1])
            event.SetId(2 - (k[0] == "Foot Strike"))
            event.SetFrame(int(frame))
            event.SetTime(frame/fps)
            acq.AppendEvent(event)

    writer = btk.btkAcquisitionFileWriter()
    writer.SetInput(acq)
    writer.SetFilename(filename_out)
    writer.Update()
    return


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
    directory = args.directory

    if not directory:
        c3dfile = getFilePath('Select c3d file to analyze').name
        c3dfile_out = c3dfile[:-4] + '_label.c3d'
        process(c3dfile, c3dfile_out, sacr=False, debug=False)
    else:
        c3ddir = getDirectoryPath('Select results directory')
        for c3dfile in os.listdir(c3ddir):
            c3dfile = c3ddir + "/" + c3dfile
            c3dfile_out = c3dfile[:-4] + '_label.c3d'
            if c3dfile.endswith(".c3d"):
                process(c3dfile, c3dfile_out, sacr=True, debug=False)
                continue
            else:
                continue
