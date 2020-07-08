import librosa
import numpy as np
import os

from Utils import Utils


class SpectogramGenerator:

    number_of_mels = 128  # The number of mel filters, it also represents the height of the spectogram
    sampling_rate = 16000  # The sampling rate used to read the input files.
    fft_points = 1024  # The number of points used for fft.
    hop_length = 160  # The interframe length.
    min_len = (
        100  # The width of the spectogram, it also represents the size of the window.
    )

    def __init__(self, filename):
        self.filename = filename

    def get_spectograms(self, output_path):
        y, sr = librosa.load(self.filename, sr=self.sampling_rate)
        y = self.__remove_silence(y)
        window_size = self.hop_length * self.min_len
        windows = int(len(y) / window_size)
        Utils.print_info(
            "{} with {} size has {} windows having {} size.".format(
                self.filename, len(y), windows, window_size
            )
        )
        i = 0
        while i < windows:
            start = i * window_size
            window = y[start : start + window_size]
            out = self.__set_path(self.filename, output_path, i)
            self.__generate_spectogram(window, out)
            i = i + 1

    @classmethod
    def __generate_spectogram(cls, y, out_prefix):
        mels = librosa.feature.melspectrogram(
            y,
            cls.sampling_rate,
            n_fft=cls.fft_points,
            n_mels=cls.number_of_mels,
            hop_length=cls.hop_length,
        )
        mels = np.log(mels + 1e-9)  # Adding a small value to avoid log(0).

        img = Utils.scale_minmax(mels, 0, 255).astype(
            np.uint8
        )  # Min-max scale to fit inside 8-bit range.
        img = np.flip(img, axis=0)  # Put low frequencies at the bottom in image.
        img = 255 - img  # Invert. Black means more energy.
        Utils.save_image(img, out_prefix)

    @classmethod
    def __remove_silence(cls, y, threshold=30, frame_length=1024, hop_length=256):
        Utils.print_info("Removing silence from current track...")
        non_silent_intervals = librosa.effects.split(
            y, top_db=threshold, frame_length=frame_length, hop_length=hop_length
        )
        new_y = y[non_silent_intervals[0][0] : non_silent_intervals[0][1]]
        for interval in non_silent_intervals[1:]:
            new_y = np.concatenate([new_y, y[interval[0] : interval[1]]])
        Utils.print_info("Silence successfully removed!")
        return new_y

    @staticmethod
    def __set_path(init_path, out_dir, index):
        base = os.path.basename(init_path)
        out = out_dir + base.replace("wav", "png")
        return Utils.append_id(out, index)
