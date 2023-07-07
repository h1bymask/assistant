# Speech Emotion Recognition
[github/deethereal](https://github.com/deethereal/speech-emotion-recognition)'s MSUFSR coursework
<br/>
[wandb](https://wandb.ai/deethereal/speech-emotion-recognition)  
<br/>
## To run training or inference:
### Setup:
1. Virtual environment
    - `python3 -m venv venv/`
    - `venv/pip install -r requirements.txt`
2. Docker
    - Create docker image:
     `docker build -t . speech_emotion_recognition`
    - Run container:
        * `docker compose up --build`   
        or
        * `docker run -it -v {cur_dir}:/workspace --rm --name dusha_docker speech_emotion_recognition` (preferred)

### Training
1. Download features from [DUSHA](https://github.com/salute-developers/golos/tree/master/dusha#downloads) to the current directory
2. Create `wand_token.txt` with your wandb token
3. Check correct paths in `conf/config.yaml`
4. `python train.py`

### Inference
1. Check correct paths in `conf/config.yaml`
2. Load some `.wav` files
3. `python inference.py`
