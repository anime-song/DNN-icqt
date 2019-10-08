from model import inverse_CQT
import librosa
import numpy as np


if __name__ == "__main__":
    
    model = inverse_CQT((None, 175), (None, 252), hidden_size=256, n_layer=2)
    model.load_weights("icqt-weight.h5")

    y, sr = librosa.load("test.mp3", sr=22050)

    mix = librosa.stft(y, n_fft=1024, hop_length=512, window="hamm")
    mag, phase = librosa.magphase(mix)

    target = librosa.cqt(
        y,
        sr=22050,
        hop_length=512,
        n_bins=12 * 3 * 7,
        bins_per_octave=12 * 3,
        filter_scale=0.5,
        window="hamm")

    pred = model.predict([np.expand_dims(mag[:175].T, 0), np.expand_dims(np.abs(target).T, 0)])

    concat = np.concatenate((pred[0].T, mag[175:]))

    i_d = librosa.istft(concat * phase, hop_length=512, win_length=1024, window="hamm")
    librosa.output.write_wav("test_out.wav", i_d, sr=22050)
