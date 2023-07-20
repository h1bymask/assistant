Speech to Text via VOSK


For start speech to text: python test_microphone.py --model=ru --device=1
(You can choose the language model after equals. Device number is determined based on the system)

"-f", "--filename", type:str, metavar="FILENAME",audio file to store recording to

"-d", "--device", type:int_or_str, input device (numeric ID or substring)

"-r", "--samplerate", type:int, sampling rate

"-m", "--model", type:str, language model; e.g. en-us, fr, nl; default is en-us.

<br/>
$ <code>ln -s cache/vosk/*-ru-0.22 $HOME/.cache/vosk/</code> <br/> <br/>
$ <code>python3 speechToText.py -d 0 -m ru</code> <br/>
(This downloads the model to $HOME/.cache/vosk/vosk-model-small-ru-0.22) <br/>
<br/>
$ <code>python3 --version</code> <br/>
Python 3.8.10 <br/>
<br/>
$ <code>pip freeze</code> <br/>
vosk==0.3.45 <br/>
sounddevice==0.4.6 <br/>
SoundFile==0.10.3.post1 <br/>
