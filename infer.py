import os
import time
import math
import json
import torch
import argparse
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import write, read
import scipy.signal as signal
from torchaudio import transforms
import torch.nn.functional as F
from util import mel_spectrogram, pitch_extract, AttrDict, pitch_inter, read_info
from models import Generator

MAX_WAV_VALUE = 32768.0


def fade_win(overlap):
    silence_len = overlap // 3
    fade_len = overlap - silence_len
    silence = np.zeros((silence_len), dtype=np.float64)
    linear = np.ones((silence_len), dtype=np.float64)

    # Equal power crossfade
    t = np.linspace(-1, 1, fade_len, dtype=np.float64)
    # fade_in = np.sqrt(0.5 * (1 + t))
    # fade_out = np.sqrt(0.5 * (1 - t))
    fade_in = 0.5 * (1 + t)
    fade_out = 0.5 * (1 - t)
    # Concat the silence to the fades
    fade_in = np.concatenate([silence, fade_in])
    fade_out = np.concatenate([linear, fade_out])
    return fade_in, fade_out


class AutoTune():
    def __init__(self, config_file, device, model_path):
        print(config_file)
        with open(config_file) as f:
            data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)
        self.h = h
        self.init_model( device, model_path)
        self.device = device

        self.frame_size = 64
        self.spec_hop_size = 256
        self.fade_frameLen = self.frame_size // 2
        
        self.fadewin = self.frame_size * self.h.super_hop_size
        self.fadeLen = int(self.fadewin/2)

        # self.hann_win = self._hanning_win(self.winLen)
        # self.fade_in_win = self.hann_win[0:self.fadeLen]
        # self.fade_out_win = self.hann_win[self.fadeLen:]
        self.fade_in_win, self.fade_out_win = fade_win(self.fadewin)
        self.cache = {'mel': np.array([]), 'pitch': np.array([])}
        self.history = np.array([])
        self.last_nframes = 100

    def init_model(self, device, model_path):
        
        self.model= Generator(self.h, device).to(device)
        self.model.load_state_dict(self.load_checkpoint(model_path, device)['generator'])
        self.model.eval()
#        self.model.remove_weight_norm()

    def infer(self, mel, pitch):
        sta_time=time.time()
        pitch = torch.from_numpy(pitch).reshape(1, -1).float().to(device)
        mel = torch.from_numpy(mel).float().to(self.device)
        mel = mel.unsqueeze(0).permute(0,2,1)
        with torch.no_grad():
            print(mel.shape, pitch.shape)
            audio = self.model(mel, pitch.unsqueeze(0))
        audio = audio.squeeze().cpu().numpy().astype(np.float64)
#        audio = y_g_hat.squeeze()
#        audio = audio * MAX_WAV_VALUE
#        audio = audio.cpu().numpy().astype('int16')
        print('infer rtf:', (time.time()-sta_time)/(len(audio)/self.h.super_sampling_rate))
        return audio

    def process(self, mel, pitch, last_seg=False):
        if len(self.cache['mel']):
            mel = np.vstack((self.cache['mel'], mel))
            pitch = np.hstack((self.cache['pitch'], pitch))
        if last_seg:
            len_mel = mel.shape[0]
            if len_mel % self.frame_size != 0:
                pad_size = (len_mel // self.frame_size + 1) * self.frame_size - len_mel

                mel = np.pad(mel, ((0,pad_size), (0, 0)), mode='constant', constant_values=-11.51292514801025390625)
                pitch = np.pad(pitch, (0, pad_size), 'reflect')
            ret_audio = self.infer(mel, pitch)
            ret_audio = ret_audio[:len_mel* self.h.super_hop_size]

            if len(self.history):
                ret_audio[:self.fadewin] = ret_audio[:self.fadewin] * self.fade_in_win + self.history
            self.cache = {'mel': np.array([]), 'pitch': np.array([])}
            self.history = np.array([])

            return ret_audio
        else:
            split_point = self._find_split_boundary(mel, self.frame_size, self.last_nframes)
            if split_point + self.fade_frameLen != 0:
                tmp = mel[:split_point + self.fade_frameLen]
                tmp_pitch = pitch[:split_point + self.fade_frameLen]
            else:
                tmp = mel
                tmp_pitch = pitch
            ret_audio = self.infer(tmp, tmp_pitch)
            if len(self.history):
                ret_audio[:self.fadewin] = ret_audio[:self.fadewin] * self.fade_in_win + self.history
                # ret_audio[:self.fadeLen]=ret_audio[:self.fadeLen]*self.fade_in_win
                # ret_audio[:2*self.fadeLen] = ret_audio[:2*self.fadeLen] * + self.history
            self.history = ret_audio[-self.fadewin:] * self.fade_out_win
            # self.history = ret_audio[split_point - self.fadeLen : split_point + self.fadeLen]
            # self.history[self.fadeLen:]= self.history[self.fadeLen:]*self.fade_out_win

            self.cache['mel'] = mel[split_point - self.fade_frameLen:]
            self.cache['pitch'] = pitch[split_point - self.fade_frameLen:]
        # ret_audio=ret_audio[:split_point - self.fadeLen]
        return ret_audio[:- self.fadewin]
    def process_feature(self, wav_path, speed_scale, key_shift, hop_s=0.005):
        sr, wav_data = read(wav_path)

        pitch = pitch_extract(wav_path, hop_s=hop_s, key_shift=key_shift)

        zeors_index = np.where(pitch == 0)[0]
        if 0 < len(zeors_index) < len(pitch):
            pitch = pitch_inter(pitch, left_win_size=5, right_win_size=5)
        pitch= torch.from_numpy(pitch)


        wav_data = wav_data / MAX_WAV_VALUE
        wav_data = normalize(wav_data) * 0.95
        # peak = np.abs(wav_data).max()
        # wav_data = wav_data / peak * 0.95
        wav_data = torch.FloatTensor(wav_data).to(self.device)
        mel = self.get_super_mel(wav_data.unsqueeze(0), self.h)

        max_len = max(mel.shape[2], pitch.shape[0])
        if len(pitch) < max_len: pitch = F.pad(pitch, (0, max_len - pitch.shape[0]))


        if speed_scale!=1.0:
            mel = F.interpolate(mel, scale_factor=speed_scale, mode="nearest")
            pitch = F.interpolate(pitch.reshape(1,1,-1), scale_factor=speed_scale, mode='nearest').squeeze()

        mel = mel.squeeze(0).transpose(1, 0).data.cpu().numpy()
        pitch= pitch.data.cpu().numpy()

        return mel, pitch

    def test(self, wav_path,  speed_scale, key_shift):

        mel, pitch = self.process_feature(wav_path, speed_scale, key_shift)

        win_len = 5000
        total_frames = mel.shape[0]

        total_audio = np.array([])
        count = math.ceil(total_frames / win_len)
        
        for i in range(count):
            tmp_data = mel[win_len * i: win_len * i + win_len]
            tmp_pitch = pitch[win_len * i: win_len * i + win_len]

            audio = self.process(tmp_data, tmp_pitch, i==count-1)
            if len(total_audio) == 0:
                total_audio = audio
            else:
                total_audio = np.concatenate((total_audio, audio), axis=0)
            #print('time:', len(total_audio.squeeze()) / 32000)

        total_audio = total_audio * MAX_WAV_VALUE
        total_audio = total_audio.astype('int16')
        return total_audio

    def load_checkpoint(self, filepath, device):
        print("Loading '{}'".format(filepath))
        assert os.path.isfile(filepath)
        checkpoint_dict = torch.load(filepath, map_location=device)
        return checkpoint_dict

    @staticmethod
    def get_super_mel(x, h):
        return mel_spectrogram(x, h.super_n_fft, h.num_mels, h.super_sampling_rate, h.super_hop_size, h.super_win_size,
                               h.super_fmin, h.super_fmax)

    @staticmethod
    def resample(audio, ori_sample_rate, new_sample_rate):
        transform = transforms.Resample(ori_sample_rate, new_sample_rate)
        audio = transform(audio)
        return audio

    @staticmethod
    def _hanning_win(length: int = 4801):
        hanning_win = signal.windows.hann(length)
        return hanning_win

    @staticmethod
    def _find_split_boundary(data: np.ndarray, frame_size: int, last_nframes: int):
        last_nframes = min(last_nframes, int(0.64 * len(data)))
        tmp = data[-last_nframes * frame_size:]
        tmp = tmp[::-1]
        rms_list = []
        for i in range(0, math.floor(len(tmp) / frame_size)):
            rms_list.append((tmp[i * frame_size:(i + 1) * frame_size] ** 2).mean())

        split_point = int((np.argmin(rms_list) + 0.5) * (-1) * frame_size)

        return split_point


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', '-c', required=True)
    parser.add_argument('--config', '-cf', required=True)
    parser.add_argument('--input_wavs_dir', '-i', default='test')
    parser.add_argument('--output_dir', '-o', default='output')
    parser.add_argument('--speed_scale', '-s', type=float, default=1.0)
    parser.add_argument('--key_shift', '-k', type=int, default=0)


    parser.add_argument('--gpu', type=int, default=0, required=False)
    parser.add_argument('--cpu', action='store_true', required=False)

    args = parser.parse_args()

    seed = 100
    torch.manual_seed(seed)
    global device

    if args.cpu:
        device = torch.device('cpu')
    else:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:{}'.format(args.gpu))

    infer = AutoTune(args.config, device, args.ckpt)

    filelist = os.listdir(args.input_wavs_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for i, filename in enumerate(filelist):
        if not filename.endswith('.wav'): continue
        wav_path = os.path.join(args.input_wavs_dir, filename)

        print(wav_path)

        output = infer.test(wav_path, args.speed_scale, args.key_shift)

        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir,
                                   os.path.splitext(filename)[0] + f'_key{args.key_shift}_speed{args.speed_scale}_generated.wav' )
        write(output_file, infer.h.super_sampling_rate, output)


if __name__ == '__main__':
    main()

