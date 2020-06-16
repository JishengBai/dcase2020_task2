"""
 @file   common.py
 @brief  Commonly used script
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
import glob
import argparse
import sys
import os

# additional
import numpy
import numpy as np
import librosa
import librosa.core
import librosa.feature
import math
import scipy.sparse as ss
import yaml
#import openl3
#import soundfile as sf

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"
########################################################################


########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2020 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.eval ^ args.dev:
        if args.dev:
            flag = True
        else:
            flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag
########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################


########################################################################
# file I/O
########################################################################
# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################
def linear_frequency(fs, NbCh, nfft, warp, fhigh, flow):
    warp = 1
    fhigh = fs / 2
    flow = 0
    LowMel = flow
    NyqMel = fhigh

    StartMel = LowMel + np.linspace(0, NbCh - 1, NbCh) / (NbCh + 1) * (NyqMel - LowMel)
    fCen = StartMel
    StartBin = np.round(nfft / fs * fCen) + 1

    EndMel = LowMel + np.linspace(2, NbCh + 1, NbCh) / (NbCh + 1) * (NyqMel - LowMel)
    EndBin = np.round(warp * nfft / fs * EndMel)

    TotLen = EndBin - StartBin + 1

    LowLen = np.append(StartBin[1:NbCh], EndBin[NbCh - 2]) - StartBin + 1
    HiLen = TotLen - LowLen + 1

    M = ss.lil_matrix((math.ceil(warp * nfft / 2 + 1), NbCh)).toarray()
    for i in range(NbCh):
        # print(M[int(StartBin[i]-1):int(StartBin[i] + LowLen[i] - 1), i].shape)
        # print(((np.linspace(1, LowLen[i], LowLen[i])).T / LowLen[i]).reshape(-1,1).shape)
        M[int(StartBin[i]-1):int(StartBin[i] + LowLen[i] - 1), i] = ((np.linspace(1, int(LowLen[i]), int(LowLen[i]))).T / LowLen[i])
        M[int(EndBin[i] - HiLen[i]):int(EndBin[i]), i] = ((np.linspace(1, int(HiLen[i]), int(HiLen[i]))[::-1]).T / HiLen[i])
    Mfull = M
    M = M[0:int(nfft / 2+1),:]
    return M, Mfull

def gen_linear_features(data,bands_num,sr):

    stft_matric = librosa.core.stft(data,n_fft=1024,hop_length=512,win_length=1024,window='hann')
    linear_W , _ = linear_frequency(fs=sr,NbCh=bands_num,nfft=1024,warp=1,fhigh=sr/2,flow=50)
    linear_spec = np.dot(linear_W.T,np.abs(stft_matric)**2)
    

    return linear_spec
    
def gen_hpss_features(data,n_fft,hop_length,mode):
    data = librosa.core.stft(data,n_fft=n_fft,hop_length=hop_length,win_length=n_fft,window='hann')
    data_h,data_p = librosa.decompose.hpss(data)
    if mode=='h':
        return np.abs(data_h)**2
    if mode=='p':
        return np.abs(data_p)**2

########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    ########
#    mel_spectrogram = librosa.feature.melspectrogram(y=y,
#                                                     sr=sr,
#                                                     n_fft=n_fft,
#                                                     hop_length=hop_length,
#                                                     n_mels=n_mels,
#                                                     power=power)
#
#    # 03 convert melspectrogram to log mel energy
#    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
#
#    # 04 calculate total vector size
#    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
#
#    # 05 skip too short clips
#    if vector_array_size < 1:
#        return numpy.empty((0, dims))
#
#    # 06 generate feature vectors by concatenating multiframes
#    vector_array = numpy.zeros((vector_array_size, dims))
#    for t in range(frames):
#        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T
      #############
#    linear_spectrogram = gen_linear_features(data=y,bands_num=n_mels,sr=sr)
#    log_linear_spectrogram = 20.0 / power * numpy.log10(linear_spectrogram + sys.float_info.epsilon)
#    vector_array_size = len(log_linear_spectrogram[0, :]) - frames + 1
#    if math.isnan(np.max(linear_spectrogram)):
#        print(file_name)
#    if vector_array_size < 1:
#        return numpy.empty((0, dims))
#    vector_array = numpy.zeros((vector_array_size, dims))
#    for t in range(frames):
#        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_linear_spectrogram[:, t: t + vector_array_size].T
    ##################
#    data_h = gen_hpss_features(data=y,n_fft=n_fft,hop_length=hop_length,mode='h')
#    if math.isnan(np.max(data_h)):
#        print(file_name)    
#    data_h_spectrogram = 20.0 / power * numpy.log10(data_h + sys.float_info.epsilon)    
#    vector_array_size = len(data_h_spectrogram[0, :]) - frames + 1
#    if vector_array_size < 1:
#        return numpy.empty((0, dims))
#    vector_array = numpy.zeros((vector_array_size, dims))
#    for t in range(frames):
#        vector_array[:, n_mels * t: n_mels * (t + 1)] = data_h_spectrogram[:, t: t + vector_array_size].T
   ###############
#    data_p = gen_hpss_features(data=y,n_fft=n_fft,hop_length=hop_length,mode='p')
#    if math.isnan(np.max(data_p)):
#        print(file_name)    
#    data_p_spectrogram = 20.0 / power * numpy.log10(data_p + sys.float_info.epsilon)    
#    vector_array_size = len(data_p_spectrogram[0, :]) - frames + 1
#    if vector_array_size < 1:
#        return numpy.empty((0, dims))
#    vector_array = numpy.zeros((vector_array_size, dims))
#    for t in range(frames):
#        vector_array[:, n_mels * t: n_mels * (t + 1)] = data_p_spectrogram[:, t: t + vector_array_size].T
    #########
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
    mfcc_fea = librosa.feature.mfcc(S=log_mel_spectrogram,n_mfcc=128)
    # 04 calculate total vector size
    vector_array_size = len(mfcc_fea[0, :]) - frames + 1
    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = mfcc_fea[:, t: t + vector_array_size].T
    return vector_array


# load dataset
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
        dirs = sorted(glob.glob(dir_path))
    else:
        logger.info("load_directory <- evaluation")
        dir_path = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
        dirs = sorted(glob.glob(dir_path))
    return dirs

########################################################################

