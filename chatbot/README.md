NN NLP<br/>
<b>Chat Bot</b><br/>
Decision Tree<br/>
Emergency chat bot (TensorFlow LSTM): Fall 2022<br/>
Gagarin chat bot (BERT transformers, torch max &amp; softmax): Summer 2023<br/>
<br/>
<hr/>
<br/><br/>
Commands to run the emergency chatbot inference:<br/><br/>
`cd chatbot`<br/>
`python3 --version` &nbsp; <i># Works fine on 3.8.10</i><br/>
`python3 -m venv venv/`<br/>
`venv/bin/python -m pip install tensorflow==2.8`<br/>
`venv/bin/python -m pip install keras==2.7.0` &nbsp; <i># Should stay incompatible with tensorflow-2.8 in order to invoke load_model, see model_was_saved_here py file</i><br/>
`venv/bin/python -m pip install protobuf==3.20`<br/>
`venv/bin/python -m pip install navec stop_words scipy`<br/>
`wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar`<br/>
`ln -s navec_hudlit_v1_12B_500K_300d_100q.tar navec_lib.tar`<br/>
venv/bin/python inference.py<br/>
<br/><br/><br/>
Additional commands to run Gagarin demo chatbot:<br/><br/>
`venv/bin/python -m pip install nltk transformers torch`<br/>
<i>Unpack `model_weight_2/pytorch_model.bin` from `Архив WinRAR.rar`</i><br/>
`echo -e "import nltk\nnltk.download('stopwords')" | venv/bin/python` &nbsp; <i># Gets downloaded to $HOME/nltk_data/corpora</i><br/>
`mkdir --parents venv/nltk_data/corpora`<br/>
`mv $HOME/nltk_data/corpora/stopwords* venv/nltk_data/corpora`<br/>
`venv/bin/python gagarin_chatbot_inference.py`</br>
