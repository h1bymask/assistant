#!/bin/bash
#
microphone_device_number=0
echo "TODO: speech-emotion-recognition/inference.py by WAV of dump_fn after stt/speechToText.py:91 gets occured each time (TensorFlow throws an error of tensor shape 5 vs 4)"
cd "$(dirname "$(readlink -f "$0")")"/../../voice/gagarin    #"
pwd
nohup python3 -m http.server >/dev/null &
server=$!
trap "echo; echo Stopping server - PID $server; kill $server" 0 1 2 3 4 5 6 8 11 13 14 15
sleep 3
if ! ps -q "$server" -o pid=; then
    echo ERROR: http.server is not running >&2
    exit 1
fi
nohup firefox https://tsniimash.ru/about/60-let-polyetu-yu-a-gagarina/peregovory-yu-a-gagarina-s-punktami-upravleniya/ >/dev/null &
nohup firefox http://localhost:8000/demo.html >/dev/null &
python3 ../../stt/speechToText.py -d $microphone_device_number -m ru | (cd ../../chatbot && venv/bin/python gagarin_chatbot_inference.py) | python3 ./Proga.py || exit 2
