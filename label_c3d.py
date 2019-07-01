import numpy as np
import urllib
import sys
import btk
import os
from FileSelectGui import *
import argparse
import pandas as pd
import tkFileDialog as fd
import detect_peaks
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt



def derivative(traj, nframes):
    traj_der = traj[1:nframes,:] - traj[0:(nframes-1),:]
    return np.append(traj_der, [[0,0,0]], axis=0)


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def get_diff(arr, fs=100):
    dt = 1/fs
    df = pd.DataFrame(arr)
    dx_v = df.diff()

    df_diff = dx_v.divide(dt, axis=0)
    return df_diff



def extract_kinematics(leg, filename_in, sacr=False):
    # Open c3d and read data
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename_in)
    reader.Update()
    acq = reader.GetOutput()
    nframes = acq.GetPointFrameNumber()
    first_frame = acq.GetFirstFrame()

    metadata = acq.GetMetaData()

    # We extract only kinematics
    kinematics = ["HipAngles", "KneeAngles", "AnkleAngles", "PelvisAngles"]
    kinematics_col = ["LHipAngles.X", "LHipAngles.Y", "LHipAngles.Z", "RHipAngles.X", "RHipAngles.Y", "RHipAngles.Z",
                      "LKneeAngles.X", "LKneeAngles.Y", "LKneeAngles.Z", "RKneeAngles.X", "RKneeAngles.Y", "RKneeAngles.Z",
                      "LAnkleAngles.X", "LAnkleAngles.Y", "LAnkleAngles.Z", "RAnkleAngles.X", "RAnkleAngles.Y", "RAnkleAngles.Z",
                      "LPelvisAngles.X", "LPelvisAngles.Y", "LPelvisAngles.Z", "RPelvisAngles.X", "RPelvisAngles.Y", "RPelvisAngles.Z"
                      ]
    markers = ["TOE", "HEE"]
    markers_col = ["LTOE.X", "LTOE.Y", "LTOE.Z", "RTOE.X", "RTOE.Y", "RTOE.Z",
                   "LHEE.X", "LHEE.Y", "LHEE.Z", "RHEE.X", "RHEE.Y", "RHEE.Z"]

    # Cols
    # 2 * 5 * 3 = 30  kinematics
    # 2 * 5 * 3 = 30  marker trajectories
    # 2 * 5 * 3 = 30  marker trajectory derivatives
    # 3 * 3 = 9       extra trajectories

    # Check if there are any kinematics in the file
    brk = True
    for point in btk.Iterate(acq.GetPoints()):
        if point.GetLabel() == "L" + kinematics[0]:
            brk = False
            break

    if brk:
        print("No kinematics in %s!" % (filename,))
        return

    # Combine kinematics into one big array
    opposite = {'L': 'R', 'R': 'L'}
    angles = [None] * (len(kinematics) * 2)
    for i, v in enumerate(kinematics):
        point = acq.GetPoint(leg + v)
        angles[i] = point.GetValues()
        point = acq.GetPoint(opposite[leg] + v)
        angles[len(kinematics) + i] = point.GetValues()

    # Get the pelvis (if no SACR marker)
    if not sacr:
        LPSI = acq.GetPoint("LPSI").GetValues()
        RPSI = acq.GetPoint("RPSI").GetValues()
        midPSI = (LPSI + RPSI) / 2
        SACR = midPSI

    if sacr:
        SACR = acq.GetPoint("SACR").GetValues()
    # incrementX = 1 if midASI[100][0] > midASI[0][0] else -1

    pos = [None] * (len(markers) * 2)
    pos_sacr = [None] * (len(markers) * 2)
    for j, w in enumerate(markers):
        point = acq.GetPoint(leg + w)
        pos[j] = point.GetValues()
        pos_sacr[j] = point.GetValues() - SACR
        point = acq.GetPoint(opposite[leg] + w)
        pos[len(markers) + j] = point.GetValues()
        pos_sacr[len(markers) + j] = point.GetValues() - SACR

    # Low pass filter
    angles_lp = butter_lowpass_filter(np.hstack(angles), cutoff=2, fs=100)
    markers_sacr_lp = butter_lowpass_filter(np.hstack(pos_sacr), cutoff=2, fs=100)
    markers_lp = butter_lowpass_filter(np.hstack(pos), cutoff=2, fs=100)
    markers_uf = np.hstack(pos)

    df_angles = pd.DataFrame(angles_lp, columns=kinematics_col)
    df_markers_sacr = pd.DataFrame(markers_sacr_lp, columns=markers_col)
    df_markers = pd.DataFrame(markers_lp, columns=markers_col)
    df_markers_uf = pd.DataFrame(markers_uf, columns=markers_col)

    return df_angles, df_markers_sacr, df_markers, df_markers_uf, first_frame


def vel_method(heel, toe, ff, sacr=False):
    if sacr:
        heel = heel.where(abs(heel)>400, other=0)
        heel = heel.values
    else:
        heel = heel.where(abs(heel) > 20, other=0)
        heel = heel.values
        toe = toe.where(abs(toe) > 20, other=0)
        toe = toe.values

    vel = np.diff(heel)
    x = np.sign(vel)

    vel2 = np.diff(toe)
    x2 = np.sign(vel2)

    if sacr:
        indexes_fs = np.where(-2 == x[1:] - x[:-1])[0] + ff
        indexes_fo = np.where(2 == x[1:] - x[:-1])[0] + ff
    else:
        indexes_fs = np.where(2 == x[1:] - x[:-1])[0] + ff
        indexes_fo = np.where(2 == x2[1:] - x2[:-1])[0] + ff
    return indexes_fs.astype(np.int32), indexes_fo.astype(np.int32)


def coord_method(heel, toe, ff, sacr=False, own=False):
    if not own:
        if sacr:
            indexes_fs = detect_peaks.detect_peaks(heel, mpd=15, mph=15) + ff
            indexes_fo = detect_peaks.detect_peaks(-toe, mpd=15) + ff
        else:
            indexes_fs = detect_peaks.detect_peaks(-heel, mpd=15) + ff
            indexes_fo = detect_peaks.detect_peaks(-toe, mpd=15) + ff
    else:
        vel = np.diff(toe)
        indexes_fs = detect_peaks.detect_peaks(-vel) + ff
        indexes_fo = detect_peaks.detect_peaks(vel) + ff

    return indexes_fs.astype(np.int32), indexes_fo.astype(np.int32)


def process(filename_in, filename_out, lstm=False, vel=False, coord=False, sacr=False, own=False, debug=True):
    filename_in = filename_in.encode()
    filename_out = filename_out.encode()

    # Get data
    df_angles, df_markers_sacr, df_markers, df_markers_uf, first_frame = extract_kinematics('L', filename_in, sacr=sacr)

    events = {}

    if lstm:
        events[("Foot Strike", "Left")] = neural_method(XR, modelFO)
        events[("Foot Strike", "Right")] = neural_method(XL, modelFO)
        events[("Foot Off", "Left")] = neural_method(XR, modelHS)
        events[("Foot Off", "Right")] = neural_method(XL, modelHS)

    # Foot-Sacrum velocity and displacement methods (Zeni et al)
    # Bad performance due to pelvis rotation in turning and pelvis displacement pre-FOG
    if sacr:
        if vel:
            events[("Foot Strike", "Left")], events[("Foot Off", "Left")] = vel_method(df_markers_sacr['LHEE.Z'], first_frame, sacr)
            events[("Foot Strike", "Right")], events[("Foot Off", "Right")] = vel_method(df_markers_sacr['RHEE.Z'], first_frame, sacr)

        if coord:
            events[("Foot Strike", "Left")], events[("Foot Off", "Left")] = coord_method(df_markers_sacr['LHEE.Z'], df_markers_sacr['LTOE.Z'], first_frame)
            events[("Foot Strike", "Right")], events[("Foot Off", "Right")] = coord_method(df_markers_sacr['RHEE.Z'], df_markers_sacr['RTOE.Z'], first_frame)
    # Foot vertical displacement (Alton et al) and vertical velocity (Schache et al)
    # Bad performance, still need to debug
    # Own idea: toe vertical velocity with peak detection
    # Heel strike = minimum of toe vertical velocity, toe off = maximum of toe vertical velocity
    else:
        if vel:
            events[("Foot Strike", "Left")], events[("Foot Off", "Left")] = vel_method(df_markers['LHEE.Z'], df_markers['LTOE.Z'], first_frame, sacr)
            events[("Foot Strike", "Right")], events[("Foot Off", "Right")] = vel_method(df_markers['RHEE.Z'], df_markers['RTOE.Z'], first_frame, sacr)
        if coord:
            events[("Foot Strike", "Left")], events[("Foot Off", "Left")] = coord_method(df_markers['LHEE.Z'], df_markers['LTOE.Z'], first_frame)
            events[("Foot Strike", "Right")], events[("Foot Off", "Right")] = coord_method(df_markers['RHEE.Z'], df_markers['RTOE.Z'], first_frame)
        if own:
            events[("Foot Strike", "Left")], events[("Foot Off", "Left")] = coord_method(df_markers['LHEE.Z'], df_markers['LTOE.Z'], first_frame, sacr, own)
            events[("Foot Strike", "Right")], events[("Foot Off", "Right")] = coord_method(df_markers['RHEE.Z'], df_markers['RTOE.Z'], first_frame, sacr, own)

    fs_l = events[("Foot Strike", "Left")]
    fo_l = events[("Foot Off", "Left")]
    fs_r = events[("Foot Strike", "Right")]
    fo_r = events[("Foot Off", "Right")]
    # Generate an interactive plot for easy labelling

    if vel:
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot((df_markers_uf['LHEE.Z']))
        [ax1.axvline(x=x, color='red') for x in fs_l]
        [ax1.axvline(x=x, color='green') for x in fo_l]
        ax1.set_title('LHEE wrt Pelvis')
        #ax1.set_ylim(-910, -850)

        ax2.plot(df_markers_uf['RHEE.Z'])
        [ax2.axvline(x=x, color='red') for x in fs_r]
        [ax2.axvline(x=x, color='green') for x in fo_r]
        ax2.set_title('RHEE wrt Pelvis')
        #ax2.set_ylim(-900, -700)

    if coord:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        ax1.plot((df_markers_uf['LHEE.Z']))
        [ax1.axvline(x=x, color='red') for x in fs_l]
        ax1.set_title('LHEE')
        # ax1.set_ylim(-910, -850)
        ax2.plot(df_markers_uf['LTOE.Z'])
        [ax2.axvline(x=x, color='red') for x in fo_l]
        ax2.set_title('LTOE')
        # axs[0, 1].set_ylim(-910, -850)
        ax3.plot(df_markers_uf['RHEE.Z'])
        [ax3.axvline(x=x, color='red') for x in fs_r]
        ax3.set_title('RHEE')
        # ax2.set_ylim(-900, -700)
        ax4.plot(df_markers_uf['RTOE.Z'])
        [ax4.axvline(x=x, color='red') for x in fo_r]
        ax4.set_title('RTOE')
        # axs[1, 1].set_ylim(-900, -700)

    if own:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        #ax1.plot((df_markers_uf['LHEE.Z']))
        ax1.plot((df_markers['LTOE.Z'].diff()))
        [ax1.axvline(x=x, color='red') for x in fs_l]
        [ax2.axvline(x=x, color='green') for x in fo_r]
        ax1.set_title('LHEE')
        # ax1.set_ylim(-910, -850)
        #ax2.plot(df_markers_uf['LTOE.Z'])
        ax2.plot((df_markers['LTOE.Z'].diff()))
        [ax2.axvline(x=x, color='green') for x in fo_r]
        ax2.set_title('LTOE')
        # axs[0, 1].set_ylim(-910, -850)
        #ax3.plot(df_markers_uf['RHEE.Z'])
        ax3.plot((df_markers['RTOE.Z'].diff()))
        [ax3.axvline(x=x, color='red') for x in fs_r]
        ax3.set_title('RHEE')
        # ax2.set_ylim(-900, -700)
        #ax4.plot(df_markers_uf['RTOE.Z'])
        ax4.plot((df_markers['RTOE.Z'].diff()))
        [ax4.axvline(x=x, color='green') for x in fo_r]
        ax4.set_title('RTOE')
        # axs[1, 1].set_ylim(-900, -700)

    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    #plt.show()

    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename_in)
    reader.Update()
    acq = reader.GetOutput()
    first_frame = acq.GetFirstFrame()
    acq.ClearEvents()
    print(first_frame)

    for k, v in events.items():
        for frame in v:
            fps = 100.0
            event = btk.btkEvent()
            event.SetLabel(k[0])
            event.SetContext(k[1])
            event.SetId(2 - (k[0] == "Foot Strike"))
            event.SetFrame(np.round(frame))
            event.SetTime(frame / fps)
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
    save = args.save
    directory = args.directory

    if save:
        prefix_results = getDirectoryPath('Select results directory')

    if not directory:
        c3dfile = getFilePath('Select c3d file to analyze').name
        c3dfile_out = c3dfile[:-4] + '_label.c3d'
        process(c3dfile, c3dfile_out, coord=False, vel=False, sacr=False, own=True)
    else:
        c3ddir = getDirectoryPath('Select results directory')
        for c3dfile in os.listdir(c3ddir):
            if c3dfile.endswith(".c3d"):
                # print(os.path.join(directory, filename))
                process(c3dfile)
                continue
            else:
                continue
