import numpy as np
import array
from pydub import AudioSegment
import torch


def torch_to_bytes(tensor, sample_width=2, fr=16000, ch=1):
    '''
    Input : torch tensor of processed audio
    Output : raw bytes of audio
    '''
    np_array = tensor.numpy().copy()
    np_array = np_array * 32768
    seg_array = array.array('h', np_array.squeeze())
    new_audio_segment = AudioSegment(seg_array, sample_width=sample_width, frame_rate=fr, channels=ch)
    return new_audio_segment.raw_data


def bytes_to_torch(args, sample_width=2, fr=16000, ch=1):
    '''
    Input : raw bytes of audio
    Output : torch tensor
    '''
    audio = AudioSegment(args.data, sample_width=sample_width, frame_rate=fr, channels=ch)
    np_array = np.array([[audio.get_array_of_samples()]], dtype='float32')
    np_array = np_array / 32768
    torch_tensor = torch.from_numpy(np_array)
    return torch_tensor

