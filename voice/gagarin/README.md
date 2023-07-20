Run: `Proga.py`<br/>
See changes: `watch -n1 demo.json`<br/>
Boomerang video: `ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -filter_complex "[0]reverse[r];[0][r]concat=n=2:v=1:a=0" 000.WAIT.mp4`<br/>
Wav2Lip: https://bhaasha.iiit.ac.in/lipsync/uploader
