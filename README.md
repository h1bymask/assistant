<b>Emotional Virtual Assistant</b>

TensorFlow
LSTM NN NLP
Chat Bot
Decision Tree
Fall 2022

Commands to run the chatbot inference:
cd chatbot
python3 -m venv venv/
venv/bin/python -m pip install tensorflow==2.8
venv/bin/python -m pip install keras==2.7.0  # Should stay incompatible with tensorflow-2.8 in order to invoke load_model, see model_was_saved_here py file
venv/bin/python -m pip install navec stop_words scipy
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
ln -s navec_hudlit_v1_12B_500K_300d_100q.tar navec_lib.tar
venv/bin/python inference.py
