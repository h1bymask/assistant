# Speech Emotion Recognition
[wandb](https://wandb.ai/motley-crew/speech-emotion-recognition?workspace=user-deethereal)  
## To run training or inference:
### Common:
  2. Create docker image `docker build -t . speech_emotion_recognition`
3. Run container:
    * `docker compose up --build`   
    or
    * `docker run -it -v {cur_dir}:/workspace --rm  --name dusha_docker speech_emotion_recognition` (preferred)

### Training
1. Download features from [DUSHA](https://github.com/salute-developers/golos/tree/master/dusha#downloads) to the current directory
2. Create `wand_token.txt` with your wandb token
1. Check correct paths in `conf/config.yaml`
3. `python train.py`
### Inference
1. Check correct paths in `conf/config.yaml`
2. Load some `.wav` files
3. `python inference.py`
