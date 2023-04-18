# Speech Emotion Recognition
[wandb](https://wandb.ai/motley-crew/speech-emotion-recognition?workspace=user-deethereal)  
To run experiment:  
1. Download features from [DUSHA](https://github.com/salute-developers/golos/tree/master/dusha#downloads) to the current directory
2. Create `wand_token.txt` with your wandb token
3. Check correct paths in `conf/config.yaml`  
4. Create docker image `docker build -t . speech_emotion_recognition`
4. Run training `docker compose up --build`

