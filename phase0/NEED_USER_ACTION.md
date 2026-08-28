# NEED USER ACTION — Phase 0.5 / 1A

## 1. Install/activate ESP-IDF (blocks required ESP32-S3 build)

Current checks: `idf.py`, CMake, Ninja, `C:\Espressif`, `%USERPROFILE%\esp`, and
`%USERPROFILE%\.espressif` are all absent.

Fastest official Windows path:

1. Run `winget install Espressif.EIM` (administrator/network approval may appear).
2. Open **Espressif Installation Manager**.
3. Choose **Easy Installation** and install the latest stable ESP-IDF (v6.0.1 is current;
   ESP-IDF v5.5.x is also acceptable).
4. Open the installed **IDF PowerShell/Terminal** and run `where idf.py`.
5. Send the displayed path here (or simply reply `ESP-IDF installed`).

Do not flash hardware yet. The immediate acceptance target is only `set-target` + `build`.

Official guide:
https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/windows-setup-update.html

## 2. Later: microphone + listening confirmation

Once the automated tests are complete, a 30-second human test is needed:

- allow microphone access;
- say “你好，青小甲”;
- listen once to `assets/i_am_awake.wav` and confirm it says “我醒来了”.

The Python microphone package `sounddevice` is not currently installed. Installation will
be requested only when the rest of the demo is ready.

## 3. Configure the local Git identity (blocks Phase 2A baseline commit)

The repository is initialized and the reviewed baseline files are staged, but neither
`user.name` nor `user.email` is configured. From the project root, run the following with
your real name and email. These commands modify this repository only, not global Git
configuration:

```powershell
git -c safe.directory=F:/ZJU_intership/task/4/WakeWord-Studio config --local user.name "YOUR NAME"
git -c safe.directory=F:/ZJU_intership/task/4/WakeWord-Studio config --local user.email "YOUR EMAIL"
```

Then reply `Git identity configured`. Codex will verify the local values, stage this
updated action file, create the baseline commit, and only then validate the second TTS
source.
