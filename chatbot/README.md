NN NLP<br/>
<b>Chat Bot</b><br/>
Decision Tree<br/>
Emergency chat bot (TensorFlow LSTM): Fall 2022<br/>
Gagarin chat bot (BERT transformers, torch max &amp; softmax): Summer 2023<br/>
<br/>
<hr/>
<br/><br/>
<span>Commands to run the emergency chatbot inference:</span><br/><br/>
<code>cd chatbot</code><br/>
<code>python3 --version</code> &nbsp; <i># Works fine on 3.8.10</i><br/>
<code>python3 -m venv venv/</code><br/>
<code>venv/bin/python -m pip install tensorflow==2.8</code><br/>
<code>venv/bin/python -m pip install keras==2.7.0</code> &nbsp; <i># Should stay incompatible with tensorflow-2.8 in order to invoke load_model, see model_was_saved_here py file</i><br/>
<code>venv/bin/python -m pip install protobuf==3.20</code><br/>
<code>venv/bin/python -m pip install navec stop_words scipy</code><br/>
<code>wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar</code><br/>
<code>ln -s navec_hudlit_v1_12B_500K_300d_100q.tar navec_lib.tar</code><br/>
<code>venv/bin/python inference.py</code><br/>
<br/><br/><br/><br/><br/><br/><br/>
<span id="bert">Additional commands to run Gagarin demo chatbot:</span><br/><br/>
<code>venv/bin/python -m pip install nltk transformers torch</code><br/>
<i>Unpack <code>model_weight_2/pytorch_model.bin</code> from <code>Архив WinRAR.rar</code></i><br/>
<code>echo -e "import nltk\nnltk.download('stopwords')" | venv/bin/python</code> &nbsp; <i># Gets downloaded to $HOME/nltk_data/corpora</i><br/>
<code>mkdir --parents venv/nltk_data/corpora</code><br/>
<code>mv $HOME/nltk_data/corpora/stopwords* venv/nltk_data/corpora</code><br/>
<code>venv/bin/python gagarin_chatbot_inference.py</code></br>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
