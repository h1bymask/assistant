Speech to Text via VOSK
For start speech to text: python test_microphone.py --model=ru --device=1
(You can choose the language model after equals. Device number is determined based on the system)

"-f", "--filename", type:str, metavar="FILENAME",audio file to store recording to

"-d", "--device", type:int_or_str, input device (numeric ID or substring)

"-r", "--samplerate", type:int, sampling rate

"-m", "--model", type=str, language model; e.g. en-us, fr, nl; default is en-us.