import os
import torch
import torch.utils.data
import numpy as np
from librosa.filters import mel as librosa_mel_fn
import parselmouth

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def pitch_extract(filename, hop_s=0.01, praat_f0_floor=75,praat_f0_ceil=1000, key_shift=0):
    voice = parselmouth.Sound(filename)
    pitch = voice.to_pitch(hop_s, praat_f0_floor, praat_f0_ceil)
    pitch_values = pitch.selected_array['frequency']
    pitch_values=pitch_values.reshape(-1)
    if key_shift == 0:
        return pitch_values

    freq_scale = 2 ** (key_shift / 12)
    pitch_values = pitch_values * freq_scale
    return pitch_values


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False,  spec_norm=True):

    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr = sampling_rate, n_fft = n_fft, n_mels = num_mels, fmin = fmin, fmax = fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')

    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    if spec_norm:
        spec = spectral_normalize_torch(spec)

    return spec

def pitch_inter(pitch, left_win_size=2, right_win_size=2):
    zeors_index= np.where(pitch==0)[0]
    zeros_bound = np.where((zeors_index[1:]-zeors_index[:-1])>1)[0]

    zeros_bound_right_index = zeors_index[zeros_bound]
    zeros_bound_left_index = zeors_index[zeros_bound+1]

    zeros_bound_left_index = np.insert(zeros_bound_left_index, 0, zeors_index[0])
    zeros_bound_right_index = np.insert(zeros_bound_right_index, len(zeros_bound_right_index), zeors_index[-1])

    for idx in range(len(zeros_bound_right_index)):
        index = zeros_bound_right_index[idx] +1
        if zeros_bound_left_index[idx]==0:
            axis_x=np.arange(index, index+right_win_size)
        elif zeros_bound_right_index[idx]==len(pitch)-1:
            axis_x_left = np.arange(zeros_bound_left_index[idx] - left_win_size, zeros_bound_left_index[idx])
            axis_x = axis_x_left

        else:
            axis_x_left = np.arange(zeros_bound_left_index[idx] - left_win_size, zeros_bound_left_index[idx])
            # axis_x_right = np.arange(index, index+right_win_size)
            axis_x = axis_x_left

        inter_x = np.arange(zeros_bound_left_index[idx], index)
        axis_y_mean = np.mean(pitch[axis_x])
        pitch[inter_x] = np.repeat(axis_y_mean, len(inter_x))
    return pitch

def read_info(info_path):
    auto_info=[]
    with open(info_path) as f:
        for line in f:
            tmp_dict = {}
            tmp_line = line.split()
            tmp_dict['st1_end1'] = [int(tmp_line[1]), int(tmp_line[2])]
            tmp_dict['st2_end2'] = [int(tmp_line[5]), int(tmp_line[6])]
            tmp_dict['sample_st1_end1'] = [int(tmp_line[3]), int(tmp_line[4])]
            tmp_dict['sample_st2_end2'] = [int(tmp_line[7]), int(tmp_line[8])]
            tmp_dict['pitch_scale'] = float(tmp_line[9])
            tmp_dict['b_speed'] = int(tmp_line[15])
            auto_info.append(tmp_dict)
    return auto_info
def read_info1(info_path):
    auto_info=[]
    with open(info_path) as f:
        for line in f:
            tmp_dict = {}
            tmp_line = line.split()
            tmp_dict['st1_end1'] = [int(tmp_line[0]), int(tmp_line[1])]
            tmp_dict['st2_end2'] = [int(tmp_line[4]), int(tmp_line[5])]
            tmp_dict['sample_st1_end1'] = [int(tmp_line[2]), int(tmp_line[3])]
            tmp_dict['sample_st2_end2'] = [int(tmp_line[6]), int(tmp_line[7])]
            tmp_dict['pitch_scale'] = float(tmp_line[8])
            auto_info.append(tmp_dict)
    return auto_info

def read_pitch(pitch_file, sr=48, hop=5):
    pitch=[]
    pitch_time =[]
    last_end_time =-1
    with open(pitch_file) as f:
        lines = [line.strip().split() for line in f.readlines()]
        for idx, line in enumerate(lines):
            if last_end_time +1 !=int(line[2]):
                # 遇到没有pitch的
                start_frame = int(round(int(last_end_time +1)/sr/hop))
                end_frame = int(round(int(lines[idx][2])/sr/hop))
                frames = np.arange(start_frame, end_frame)
                if len(pitch_time)>0 and len(frames)>0 and frames[0]-1!= pitch_time[-1]:
                    print(line)
                pitch_time.extend(frames)
                pitch.extend([0.0]*len(frames))
                # print('error:', pitch_file, last_end_time)
            start_frame = int(round(int(lines[idx][2])/sr/hop))
            end_frame = int(round(int(lines[idx][3])/sr/hop))
            frames = np.arange(start_frame, end_frame)
            if len(pitch_time)>0 and len(frames)>0 and frames[0]-1!= pitch_time[-1]:
                print(line, pitch_time[-1])
            pitch_time.extend(frames)
            pitch.extend([float(line[4])]*len(frames))


            last_end_time = int(line[3])
    pitch = np.array(pitch)

    return np.array(pitch)
