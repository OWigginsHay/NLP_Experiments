import simpleaudio as sa
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import wavfile
import numpy as np
import seaborn as sns
import math
import wave

fs = 44100

class AudioBoi:

    def _wav2array(sefl, nchannels, sampwidth, data):
        """data must be the string containing the bytes from the wav file."""
        num_samples, remainder = divmod(len(data), sampwidth * nchannels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of '
                            'sampwidth * num_channels.')
        if sampwidth > 4:
            raise ValueError("sampwidth must not be greater than 4.")

        if sampwidth == 3:
            a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
            a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
            result = a.view('<i4').reshape(a.shape[:-1])
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sampwidth == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
            result = a.reshape(-1, nchannels)
        return result

    def readwav(self, file):
        """
        Read a wav file.
        Returns the frame rate, sample width (in bytes) and a numpy array
        containing the data.
        This function does not read compressed wav files.
        """
        wav = wave.open(file)
        rate = wav.getframerate()
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        nframes = wav.getnframes()
        data = wav.readframes(nframes)
        wav.close()
        array = self._wav2array(nchannels, sampwidth, data)
        return rate, sampwidth, array

    def Gauss1D(self, N, sigma):
        out = -np.arange(-N//2+1,N//2+1)**2
        out = out/(2*sigma**2)
        out = np.exp(out)
        return out

    def Spectogram(self, audio_data, window_time):
        sample_space = math.floor(window_time*1e-3*fs/2)*2
        print(sample_space)
        sample_number = math.floor(audio_data.shape[0]/(sample_space))
        print(sample_number)
        spec_size = math.ceil(sample_space/2)
        print(spec_size)
        spec = np.zeros((spec_size, sample_number))

        for i in range(sample_number):
            slice = audio_data[i*sample_space:(i+1)*sample_space, 0]
            slice = np.fft.fft(slice)
            slice = np.fft.fftshift(slice)
            spec[:, i] = np.abs(slice[:spec_size])
            recreate = np.fft.ifft(np.fft.fftshift(slice))
        print(spec.shape)
        sns.heatmap(spec, norm=LogNorm(vmin=1e-10, vmax=100*spec.max()))
        plt.show()
        return spec

    def Record(self, seconds, save = False):

        sd.default.samplerate = fs
        sd.default.channels = 2
        print(sd.query_devices())
        print("Recording")
        local_record = sd.rec(seconds * fs)
        sd.wait()
        print("Done")
        return local_record

    def GetFileAudio(self, f, play = False):
        r,_,np_audio = self.readwav(f)
        np_audio = np_audio.astype(np.float32)
        np_audio *= (2**15-1)/np.max(np.abs(np_audio))
        np_audio = np_audio.astype(np.int16)
        if play:
            play = sa.play_buffer(np_audio, 1, 2, 2*r)
            play.wait_done()
        return np_audio
        
if __name__ == "__main__":
    ab = AudioBoi()

    ab.Spectogram(ab.GetFileAudio(
        r"F:\Users\OWiggins\Desktop\EP\Azazel.wav", True), 20)
    #ab.Spectogram(Record(10), 20)

    #Spectogram(local_record, 10)

    # sig = 10000
    # low = Gauss1D(seconds*fs, sig)
    # low[low == np.inf] = 0
    # low[:start_index-3*sig] = 0
    # low[start_index+3*sig:] = 0

    # audio = recreate * (2**15-1)/np.max(np.abs(recreate))
    # audio = audio.astype(np.int16)
    # play_obj = sa.play_buffer(audio, 1, 2, fs)
    # play_obj.wait_done()

    # filename = r"F:\Users\OWiggins\Desktop\EP\Azazel.wav"
    # wave_obj = sa.WaveObject.from_wave_file(filename)
    # play_obj = wave_obj.play()
    # play_obj.wait_done()




