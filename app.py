import streamlit as st
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential, Model, model_from_json

# Other
import librosa
import librosa.display
import json

def main():

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title('Speech Emotion Recognition')
    st.write('Created by Sandeep R :)')
    st.write('''
    # Speech Emotion Recognition''')



    speech_file = st.file_uploader(label = 'Please Upload Audio File')
    if speech_file is not None:
        st.write('File Uploaded')
        st.write('Click on Play Button to Hear Audio file')
        st.audio(speech_file, format='audio / ogg')
        test_angry = pd.DataFrame(columns=['feature'])
        path_angry = speech_file
        # loop feature extraction over the entire dataset
        counter = 0
        X, sample_rate = librosa.load(path_angry, res_type='kaiser_fast'
                                    , duration=3
                                    , sr=44100
                                    , offset=0.5
                                    )
        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128, fmax=8000)

        db_spec = librosa.power_to_db(spectrogram)
        log_spectrogram = np.mean(db_spec, axis=0)
        test_angry.loc[counter] = [log_spectrogram]
        counter = counter + 1

        # Check a few records to make sure its processed successfully
        test_angry.head()

        test_angry = pd.DataFrame(test_angry['feature'].values.tolist())
        test_angry.reset_index(drop=True, inplace=True)

        test_angry.dropna(axis=0, inplace=True)

        test_angry = np.array(test_angry)
        test_angry_new = np.append(test_angry, np.zeros(259 - test_angry.shape[1], ))
        test_angry_new = test_angry_new.reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(test_angry_new)
        test_angry = scaler.transform(test_angry_new)
        test_angry_new = np.expand_dims(test_angry_new, axis=1)
        test_angry_new = np.expand_dims(test_angry_new, axis=0)

        # loading json and model architecture
        json_file = open('model_json.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("Emotion_Model.h5")

        # predicting
        preds = loaded_model.predict(test_angry_new,
                                  verbose=1)

        preds = preds.argmax(axis=1)
        labels_dict = {1: 'angry', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}
        #st.write('Uploaded File is of Emotion:')
        st.write('Uploaded File is of Emotion:',labels_dict[int(preds)])

if __name__ == '__main__':
    main()