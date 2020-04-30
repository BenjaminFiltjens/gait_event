# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:10:28 2017

Small functions to select file or directory paths via a GUI

Functions:
    - getFilePath: Get path of a single file with a GUI
    - getDirectoryPath: Get path of a directory with a GUI

@author: Robin Amsters
@email: robin.amsters@gmail.com
"""
import numpy as np
import os

from fnmatch import fnmatch
from Tkinter import Tk
from tkFileDialog import askopenfile , askdirectory

def getFilePath(msg):
	# Selecting file trough GUI
	Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
	filePath = askopenfile(title = msg) # show an "Open" dialog box and return the path to the selected file
	return filePath

def getDirectoryPath(msg):
	# Selecting directory trough GUI
	Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
	filePath = askdirectory(title = msg) # show an "Open" dialog box and return the path
	return filePath

def getAllFilesInDirectory(directory, pattern):
    """
        Function that returns all the path to all files in a directory
        and its subdirectories that end in a certain pattern.
        
    """
    
    root = directory
    allFiles = np.array([])
    
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                allFiles = np.append(allFiles, os.path.join(path, name))
    
    return allFiles

def getFolderNames(directory):
    
    folders = os.walk(directory).next()[1]
    
    return folders


if __name__=='__main__':
    directory = getDirectoryPath('select directory')
    print(getAllFilesInDirectory(directory, '*.bag'))