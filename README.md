# Sonus – Acoustic Machine Fault Detection

**Hackathon project · 4th place out of 28 teams**

Sonus listens to an industrial fan and decides whether it sounds healthy or faulty. A small CNN classifies Mel spectrograms of the fan's sound. A desktop dashboard plays the audio, shows the result, and stops the motor through an Arduino when it detects a fault. It then asks Gemini to write a short technical report.

## How it works

```
fan sound (.wav) ─► first channel, 16 kHz, 3 s ─► Mel spectrogram (128 mels, dB)
        ─► resize to 128×128, min-max normalize ─► CNN ─► P(fault)
                                                            │
                         P(fault) > 0.5 ─► stop motor (Arduino "M0") + Gemini report
```

**Model.** The CNN has two Conv–ReLU–MaxPool blocks (32 and 64 filters) and two fully connected layers (128 → 1). A sigmoid output gives P(fault). The model has about 8.4M parameters, most of them in the first dense layer. It was trained with BCE loss and the Adam optimizer (lr = 0.001) for 20 epochs, with batch size 16.

**Data.** Training used a mini subset of the MIMII dataset: the fan machine type at the noisiest setting (−6 dB SNR). The subset has 400 clips, 200 normal and 200 abnormal. Training ran on a Kaggle GPU notebook.

**Dashboard.** The dashboard (CustomTkinter + Matplotlib) runs a two-step demo:
1. It plays a normal clip, draws its waveform with a moving cursor, and shows the model's score.
2. It plays an abnormal clip. If the model flags it, the app sends `M0` over serial to stop the motor and shows a "CRITICAL" state. It then asks Gemini for a short technical analysis.

If no Arduino is connected, the app keeps running in simulation mode.

## Results

| Metric | Value | Note |
|---|---|---|
| Training accuracy (epoch 20) | 96.5% | measured on the training data |
| Training loss (epoch 20) | 0.112 | BCE |
| Demo check | 6 / 6 correct | 3 normal, 3 abnormal clips |

**Limitation:** These numbers are not held-out test results. The training notebook trains on all 400 clips without a separate test split. The demo clips come from the same dataset, so they may also have been seen during training. A proper evaluation is listed under next steps.

## Repository structure

| File | Purpose |
|---|---|
| `Sonus/EgitimOrtami.ipynb` | Training notebook (Kaggle): dataset class, preprocessing, model, training loop |
| `Sonus/Uygulama.ipynb` | Desktop dashboard: model inference, Arduino control, Gemini report |
| `Sonus/fan_model_minus6db.pth` | Trained model weights (PyTorch `state_dict`) |
| `Sonus/normal.wav`, `Sonus/anormal.wav` | Example clips from MIMII (fan, −6 dB) used by the demo |

## Run the dashboard

```bash
pip install torch torchaudio librosa numpy matplotlib customtkinter sounddevice soundfile pyserial google-genai
```

1. Open `Sonus/Uygulama.ipynb` and set the values in `Config`:
   - `ARDUINO_PORT` (default `COM10`, 9600 baud)
   - `API_KEY` for Gemini. Keep this out of git, for example by reading it from an environment variable.
2. Run the cell from the `Sonus/` folder so that the model and `.wav` files are found.
3. Press **▶ SENARYOYU BAŞLAT** (start the scenario).

**Arduino protocol:** `M1\n` turns the motor on, and `M0\n` turns it off.

## Retrain the model

Open `Sonus/EgitimOrtami.ipynb` on Kaggle and attach the "MIMII mini" dataset (fan, −6 dB). Then run the cells. The notebook saves the model as `fan_model_minus6db.pth`.

## Tech stack

Python · PyTorch · torchaudio · librosa · CustomTkinter · Matplotlib · sounddevice · pySerial · Arduino · Google Gemini API · Kaggle

## Next steps

- Split the data into train, validation and test sets, then report test accuracy, precision, recall and AUC.
- Test the model at the other MIMII noise levels and on other machine types.
- Classify live microphone input instead of stored clips.
- Move the dashboard from a notebook into an `app.py` file and add a `requirements.txt`.

## Dataset and acknowledgements

This project uses audio from the **MIMII dataset** (Hitachi, Ltd.):

> H. Purohit, R. Tanabe, K. Ichige, T. Endo, Y. Nikaido, K. Suefusa, Y. Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," *DCASE 2019 Workshop*. https://zenodo.org/record/3384388

Check the dataset license on the Zenodo page before you reuse or redistribute the audio files.
