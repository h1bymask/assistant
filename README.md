<b>Emotional Virtual Assistant</b>
<br/>
TensorFlow<br/>
LSTM NN NLP<br/>
Chat Bot<br/>
Decision Tree<br/>
Fall 2022<br/>
<br/>
<hr/>
<br/>
Commands to run the chatbot inference:<br/><br/>
cd chatbot<br/>
python3 -m venv venv/<br/>
venv/bin/python -m pip install tensorflow==2.8<br/>
venv/bin/python -m pip install keras==2.7.0 &nbsp; # Should stay incompatible with tensorflow-2.8 in order to invoke load_model, see model_was_saved_here py file<br/>
venv/bin/python -m pip install protobuf==3.20<br/>
venv/bin/python -m pip install navec stop_words scipy<br/>
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar<br/>
ln -s navec_hudlit_v1_12B_500K_300d_100q.tar navec_lib.tar<br/>
venv/bin/python inference.py
