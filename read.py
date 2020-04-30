import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import re
import btk
from FileSelectGui import *
from scipy.signal import butter, filtfilt
import argparse
import random

marker = False
sacr = True


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def derivative(traj, nframes):
    traj_der = traj[1:nframes,:] - traj[0:(nframes-1),:]
    shape = [[0,0,0]*(traj.shape[1]/3)]
    return np.append(traj_der, shape, axis=0)

def extract_kinematics(leg, filename):
    m = re.match(input_dir + "(?P<name>.+).c3d", filename)
    name = m.group('name').replace(" ", "-")
    output_file = "%s/%s.csv" % (output_dir, name)
    print("Trying %s" % (filename))
    
    # Open c3d and read data
    reader = btk.btkAcquisitionFileReader() 
    reader.SetFilename(str(filename))
    reader.Update()
    acq = reader.GetOutput()
    nframes = acq.GetPointFrameNumber()

    start = acq.GetFirstFrame()
    end = acq.GetLastFrame()

    # Check if there is a FOG
    for event in btk.Iterate(acq.GetEvents()):
        if event.GetLabel() == 'FOG':
            end = event.GetFrame()
            nframes = end - start

    if os.path.isfile(output_file):
        return

    metadata = acq.GetMetaData()

    # We extract only kinematics
    kinematics = ["HipAngles", "KneeAngles", "AnkleAngles"]
    markers = ["TOE", "KNE", "HEE"]
    
    # ------------ Cols
    # If marker:
    # 2 * 3 * 3 = 18  kinematics
    # 2 * 3 * 3 = 18  marker trajectories
    # 2 * 3 * 3 = 18  marker trajectory derivatives
    # 2 * 3 * 3 = 18  kinematics derivatives
    #           = 72

    # If no markers (kin):
    # 2 * 3 * 3 = 18  kinematics
    # 2 * 3 * 3 = 18  kinematics derivatives
    #           = 36

    outputs = np.array([[0] * nframes]).T
    
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
    angles = [None] * (len(kinematics)*2)
    for i, v in enumerate(kinematics):
        point = acq.GetPoint(leg + v)
        angles[i] = point.GetValues()
        angles[i] = angles[i][:nframes]
        point = acq.GetPoint(opposite[leg] + v)
        angles[len(kinematics) + i] = point.GetValues()
        angles[len(kinematics) + i] = angles[len(kinematics) + i][:nframes]

    # Get the pelvis
    if sacr:
        SACR_X = acq.GetPoint("SACR").GetValues()[:, 0]
    else:
        LPSI_X = acq.GetPoint("LPSI").GetValues()[:, 0]
        RPSI_X = acq.GetPoint("RPSI").GetValues()[:, 0]
        midPSI_X = (LPSI_X + RPSI_X) / 2
        SACR_X = midPSI_X
    # incrementX = 1 if midASI[100][0] > midASI[0][0] else -1

    pos = [None] * (len(markers)*2)

    pos_sacr_X = [None] * (len(markers)*2)
    pos_sacr_Y = [None] * (len(markers)*2)
    pos_sacr_Z = [None] * (len(markers)*2)
    pos_sacr = [None] * (len(markers)*2)
    for j, w in enumerate(markers):
        try:
            point = acq.GetPoint(leg + w)
            pos[j] = point.GetValues()
            pos_sacr_X[j] = point.GetValues()[:, 0] - SACR_X
            pos_sacr_Y[j] = point.GetValues()[:, 1]
            pos_sacr_Z[j] = point.GetValues()[:, 2]
            pos_sacr[j] = np.column_stack((pos_sacr_X[j], pos_sacr_Y[j], pos_sacr_Z[j]))
            pos_sacr[j] = pos_sacr[j][:nframes]

            point = acq.GetPoint(opposite[leg] + w)
            pos[len(markers) + j] = point.GetValues()
            pos_sacr_X[len(markers) + j] = point.GetValues()[:, 0] - SACR_X
            pos_sacr_Y[len(markers) + j] = point.GetValues()[:, 1]
            pos_sacr_Z[len(markers) + j] = point.GetValues()[:, 2]
            pos_sacr[len(markers) + j] = np.column_stack(
                (pos_sacr_X[len(markers) + j], pos_sacr_Y[len(markers) + j], pos_sacr_Z[len(markers) + j]))
            pos_sacr[len(markers) + j] = pos_sacr[len(markers) + j][:nframes]
        except:
            return

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


    # Add events as output
    for event in btk.Iterate(acq.GetEvents()):
        if start < event.GetFrame() < end:
            if event.GetLabel() == "Foot Strike":
                if event.GetContext() == 'Left':
                    outputs[event.GetFrame() - start, 0] = 1
                elif event.GetContext() == 'Right':
                    outputs[event.GetFrame() - start, 0] = 2
            elif event.GetLabel() == "Foot Off":
                if event.GetContext() == 'Left':
                    outputs[event.GetFrame() - start, 0] = 3
                elif event.GetContext() == 'Right':
                    outputs[event.GetFrame() - start, 0] = 4
            
    if (np.sum(outputs) == 0):
        print("No events in %s!" % (filename,))
        return

    arr = np.concatenate((curves, outputs), axis=1)

    # Remove data before and after first event minus some random int. This is for those trials that are not pre-cut (Spildooren).
    if sacr:
        positives = np.where(arr[:, 36] > 0.5)
        if len(positives[0]) == 0:
            return None

        first_event = positives[0][0] - random.randint(5, 15)
        last_event = positives[0][-1] + random.randint(5, 15)
        arr = arr[first_event:last_event]

    print("Writig %s" % filename)
    np.savetxt(output_file, arr, delimiter=',')


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
        output_dir = getDirectoryPath('Select results directory')

    if not directory:
        c3dfile = getFilePath('Select c3d file to analyze').name
        c3dfile_out = c3dfile[:-4] + '_label.c3d'
    else:
        input_dir = getDirectoryPath('Select c3d input directory')
        files = os.listdir(input_dir)
        for filename in files:
            for leg in ['L', 'R']:
                filename_1 = filename.encode()
                extract_kinematics(leg, input_dir + "/" + filename_1)
                print(filename)
