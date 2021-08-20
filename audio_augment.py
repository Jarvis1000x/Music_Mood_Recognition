import librosa
import numpy as np
import config


def get_item(clusters):
    return librosa.util.find_files(config.DATA_PATH + "/" + str(clusters))


def readfile(file_name):
    y, sr = librosa.load(file_name, config.SAMPLE_RATE)
    return y, sr


def change_pitch_and_speed(file):
    y_pitch_speed = file.copy()
    length_change = np.random.uniform(low=0.8, high=1.2)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arrange(0, len(y_pitch_speed), speed_fac), np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed


def change_pitch(file, sr):
    y_pitch = file.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype("float64"), sr, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)
    return y_pitch


def value_aug(file):
    y_aug = file.copy()
    dyn_change = np.random.uniform(low=0.8, high=3)
    y_aug = y_aug * dyn_change
    return y_aug


def add_noise(file):
    noise = np.random.randn(len(file))
    data_noise = file + 0.005 * noise
    return data_noise


def hpss(file):
    y_harmonic, y_percussive = librosa.effects.hpss(file.astype("float64"))
    return y_harmonic, y_percussive


def shift(file):
    return np.roll(file, 1600)


def stretch(file, rate=1):
    input_length = len(file)
    stretching = librosa.effects.time_stretch(file, rate)
    if len(stretching) > stretching[:input_length]:
        stretching = stretching[:input_length]
    else:
        stretching = np.pad(stretching, (0, max(0, input_length - len(stretching))), "constant")
    return stretching


def change_speed(file):
    y_speed = file.copy()
    speed_change = np.random.uniform(low=0.9, high=1.1)
    tmp = librosa.effects.time_stretch(y_speed.astype("float64"), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed
















