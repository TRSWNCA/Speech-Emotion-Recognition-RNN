import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import python_speech_features as psf
import librosa
import librosa.display

emotions = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def preemphasis(signal, coeff=0.95):
    return np.append(signal[1], signal[1:] - coeff * signal[:-1])


def pow_spec(frames, NFFT):
    complex_spec = np.fft.rfft(frames, NFFT)
    return 1 / NFFT * np.square(np.abs(complex_spec))


def frame_sig(sig, frame_len, frame_step, win_func):
    '''
    :param sig: 输入的语音信号
    :param frame_len: 帧长
    :param frame_step: 帧移
    :param win_func: 窗函数
    :return: array of frames, num_frame * frame_len
    '''
    slen = len(sig)

    num_frames = 10
    frame_step = slen / num_frames
    padlen = int(num_frames * frame_step + frame_len)

    # 将信号补长，使得(slen - frame_len) /frame_step整除
    zeros = np.zeros((padlen - slen,))
    padSig = np.concatenate((sig, zeros))

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = padSig[indices]
    win = np.tile(win_func(frame_len), (num_frames, 1))
    return frames * win


def read_dataset(filepath):
    for (label, emotion) in enumerate(emotions):
        emotion_dir = filepath + '/' + emotion + '/'
        speach_files = os.listdir(emotion_dir)
        for speach_file in speach_files[:10]:
            if speach_file.split('.')[-1] == 'wav':
                y, sr = sf.read(emotion_dir + speach_file)
                # 预加重
                y = preemphasis(y, coeff=0.98)
                # 分帧加窗
                frames = frame_sig(y, frame_len=2048, frame_step=512, win_func=np.hanning)
                print(frames.shape)


read_dataset('../data/liuchanhg')
