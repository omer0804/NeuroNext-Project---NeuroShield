import mne
import matplotlib.pyplot as plt

edf_path = 'C:\\Users\\USER\\Desktop\\NeuroNext-Project---NeuroShield\\data\\Raw\\SC4001E0-PSG.edf'

raw = mne.io.read_raw_edf(edf_path, preload=True)
raw.plot()
plt.show()
