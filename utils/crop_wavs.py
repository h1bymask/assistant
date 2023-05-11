from scipy.io import wavfile
import numpy as np
import librosa 
import jsonlines
import os

from pathlib import Path
def crop_wav(originalWavPath, folder_for_wavs, train_jsonl, test_jsonl, min_len=3, max_len=8, ):
    total_length = librosa.get_duration(filename=originalWavPath)
    sampleRate, waveData = wavfile.read( originalWavPath )
    cur_ts = 0
    cntr_ = 0
    with jsonlines.open(train_jsonl, mode='a') as train_writer, jsonlines.open(test_jsonl, mode='a')  as test_writer:
        while cur_ts<total_length:
            tmp_size = (max_len-min_len)*np.random.rand() + min_len
            if total_length - cur_ts >= min_len: 
                end_ = min(cur_ts+tmp_size, total_length)
                startSample = int( cur_ts * sampleRate )
                endSample = int( end_ * sampleRate )
                pth = Path(originalWavPath)
                file_name= pth.parent.stem + '_' + pth.stem + f"_{cntr_}"
                wavfile.write(f"{folder_for_wavs}\\{file_name}.wav", sampleRate, waveData[startSample:endSample])
                if np.random.rand()<=0.2:
                    test_writer.write({'id':file_name,'tensor':f"../features/{file_name}.npy", 'wav_length':tmp_size,"label":4,'emotion':'silence'})

                else:    
                    train_writer.write({'id':file_name,'tensor':f"../features/{file_name}.npy", 'wav_length':tmp_size,"label":4,'emotion':'silence'})

                cntr_+=1
                cur_ts+=tmp_size
            else:
                break

if __name__ == '__main__':
    for folder in os.listdir('D:\\trash\\demand\\'):
        for file in os.listdir('D:\\trash\\demand\\'+folder):
            crop_wav('D:\\trash\\demand\\'+folder+'\\'+file, 
                    test_jsonl='data\\info\\test\\kursa4_test.jsonl',
                    train_jsonl='data\\info\\kursa4_data.jsonl',
                    folder_for_wavs='E:\\kursa4\\speech-emotion-recognition\\data\\silence_wavs',
                    min_len=np.random.randint(2,5),
                    max_len=np.random.randint(7,11)
                    )

        