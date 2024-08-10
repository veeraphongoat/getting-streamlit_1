import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import load_model
import librosa
import librosa.display
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import sounddevice as sd
import mysql.connector
from mysql.connector import Error
import plotly.express as px  
import pandas as pd

# Set the backend to a non-interactive one
matplotlib.use('Agg')

# Load model and label encoder
model = load_model('mymodel_3.h5')
with open('Enc_labels.sav', 'rb') as file:
    lb = pickle.load(file)
scaler = StandardScaler()

# Database connection function
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',  # your database password
            database='phuean'
        )
        if connection.is_connected():
            st.success("Connection to MySQL DB successful")
    except Error as e:
        st.error(f"The error '{e}' occurred")
    
    return connection

# Function to create table if not exists
def create_audio_table(connection):
    cursor = connection.cursor()
    query = """
    CREATE TABLE IF NOT EXISTS audio_files (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        file_name VARCHAR(255) NOT NULL,
        emotion_label TEXT NOT NULL,
        most_frequent_emotion VARCHAR(50) NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """
    cursor.execute(query)
    connection.commit()
    cursor.close()

# Function to insert audio file and emotion label into database
def insert_audio(connection, user_id, file_name, emotion_labels, most_frequent_emotion):
    cursor = connection.cursor()
    query = """
    INSERT INTO audio_files (user_id, file_name, emotion_label, most_frequent_emotion) 
    VALUES (%s, %s, %s, %s)
    """
    emotion_str = ','.join(emotion_labels)
    cursor.execute(query, (user_id, file_name, emotion_str, most_frequent_emotion))
    connection.commit()
    cursor.close()

# Feature extraction and audio processing functions
def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    zcr_feat = zcr(data, frame_length, hop_length)
    rmse_feat = rmse(data, frame_length, hop_length)
    mfcc_feat = mfcc(data, sr, frame_length, hop_length)

    max_length = 2376
    if len(mfcc_feat) < max_length:
        mfcc_feat = np.pad(mfcc_feat, (0, max_length - len(mfcc_feat)), 'constant')
    elif len(mfcc_feat) > max_length:
        mfcc_feat = mfcc_feat[:max_length]

    features = np.hstack((zcr_feat, rmse_feat, mfcc_feat))

    if len(features) < max_length:
        features = np.pad(features, (0, max_length - len(features)), 'constant')
    elif len(features) > max_length:
        features = features[:max_length]

    return features

def predict_emotion(audio_data, sr):
    segment_length = 5 * sr  # 5 seconds in samples
    num_segments = len(audio_data) // segment_length
    emotion_labels = []

    for i in range(num_segments):
        segment = audio_data[i*segment_length:(i+1)*segment_length]
        features = extract_features(segment, sr)
        features = features.reshape(1, -1, 1)  # Reshape to (1, 2376, 1) for the model

        y_pred = model.predict(features)
        predicted_class = lb.classes_[np.argmax(y_pred)]
        emotion_labels.append(predicted_class)
    
    return emotion_labels

def get_user_info(username):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        query = "SELECT id FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result:
            user_id = result[0]  # Assuming username is unique
            return user_id
        else:
            st.error(f"No user found with username: {username}")
            return None
    return None

def plot_sunburst_chart(emotion_labels):
    unique_labels, counts = np.unique(emotion_labels, return_counts=True)
    df = pd.DataFrame({
        'Emotion': unique_labels,
        'Count': counts
    })
    
    # Define custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    fig = px.sunburst(df, path=['Emotion'], values='Count', color_discrete_sequence=colors)
    
    # Update layout to center the chart
    fig.update_layout(
        height=400,
        width=400,
        margin=dict(t=20, l=20, r=20, b=20),  
        autosize=True,
        template="plotly_white"  # Use a light theme for better visibility
    )
    
    return fig

def get_image_from_db(base_name):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        query = "SELECT name, image FROM images WHERE name LIKE %s"
        cursor.execute(query, (f"{base_name}%",))  # ใช้ LIKE เพื่อค้นหาชื่อที่คล้ายกัน
        results = cursor.fetchall()
        cursor.close()
        connection.close()

        if results:
            # เลือกผลลัพธ์แรกที่พบ
            image_name, image_data = results[0]
            return image_name, image_data
        else:
            st.error(f"No image found with base name: {base_name}")
            return None, None
    return None, None

def deploy(username):
    st.markdown(
        """
        <style>
        .main {
            background-color: #F0F2F6;
        }
        .stApp {
            background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
            color: black;
        }
        .stApp header {
            background: linear-gradient(to right, violet, indigo, blue, green, yellow, orange, red);
        }
        .stApp .block-container {
            background: rgba(255, 255, 255, 0.8);
            padding: 2rem;
            border-radius: 10px;
        }
        .stApp .stButton>button {
            background: red;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
        }
        .stApp .stButton>button:hover {
            background: darkred;
        }
        h1, h2, h3, h4, h5, h6, .stApp .stMarkdown {
            color: black;
        }
        .sunburst-container {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title(f"Sentiment Classification from Audio - User: {username}")

    with st.sidebar:
        st.header("Instructions")
        st.write("""
        Upload an audio file in WAV format. The application will predict the emotion conveyed in the audio.
        """)
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

    if uploaded_file is not None:
        audio_data, sr = librosa.load(uploaded_file, sr=None)
        
        # Check duration
        duration = librosa.get_duration(y=audio_data, sr=sr)
        if duration > 60:  # 60 seconds limit
            st.error("The audio file must be 1 minute or less.")
        else:
            emotion_labels = predict_emotion(audio_data, sr)

            # Count occurrences of each emotion
            emotion_count = {}
            for emotion in emotion_labels:
                if emotion in emotion_count:
                    emotion_count[emotion] += 1
                else:
                    emotion_count[emotion] = 1

            st.subheader("Predicted Emotions for each 5-second segment:")
            for idx, emotion in enumerate(emotion_labels):
                st.write(f"Segment {idx + 1}: {emotion}")

            # Display emotion counts
            st.subheader("Emotion Counts:")
            for emotion, count in emotion_count.items():
                st.write(f"{emotion}: {count}")

            # Identify the most frequent emotion
            most_frequent_emotion = max(emotion_count, key=emotion_count.get)
            st.write(f"Most Frequent Emotion: {most_frequent_emotion}")

            # Check for and display related image
            image_base_name = most_frequent_emotion  # ใช้ชื่ออารมณ์เพื่อค้นหารูปภาพ
            image_name, image_data = get_image_from_db(image_base_name)
            if image_data:
                st.subheader("Related Image for Most Frequent Emotion:")
                st.image(BytesIO(image_data))

            # Display sunburst chart of emotion distribution
            st.subheader("Emotion Distribution Sunburst Chart:")
            st.markdown('<div class="sunburst-container">', unsafe_allow_html=True)
            sunburst_chart_fig = plot_sunburst_chart(emotion_labels)
            st.plotly_chart(sunburst_chart_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.header("Waveform")
                fig, ax = plt.subplots()
                librosa.display.waveshow(audio_data, sr=sr, color="purple", ax=ax)
                ax.set(xlabel='Time (s)', ylabel='Amplitude', title='Waveform')
                st.pyplot(fig)
                st.audio(uploaded_file)

            with col2:
                st.header("Spectrogram")
                fig, ax = plt.subplots()
                spec = librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max),
                                                sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='coolwarm')
                ax.set(title='Spectrogram')
                st.pyplot(fig)

            if st.button("Save Audio", key="save_uploaded_audio"):
                connection = create_connection()
                if connection:
                    create_audio_table(connection)
                    user_id = get_user_info(username)
                    if user_id:
                        # Save uploaded file to database
                        file_bytes = uploaded_file.read()
                        file_name = uploaded_file.name
                        try:
                            insert_audio(connection, user_id, file_name, emotion_labels, most_frequent_emotion)
                            st.success("Uploaded audio file saved to MySQL database.")
                        except Exception as e:
                            st.error(f"Error saving uploaded audio: {e}")
                    else:
                        st.error("User ID not found.")
                    connection.close()

            
    st.header("Record Audio")
    duration = st.number_input("Enter duration in seconds", min_value=1, max_value=60, value=5)
    sample_rate = 44100  # Sample rate in Hz

    if st.button("Record"):
        st.write("Recording...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()  # Wait until recording is finished
        st.write("Recording finished")

        audio_data = recording.flatten()
        sr = sample_rate
        emotion_labels = predict_emotion(audio_data, sr)

        st.subheader("Predicted Emotions for each 5-second segment:")
        for idx, emotion in enumerate(emotion_labels):
            st.write(f"Segment {idx + 1}: {emotion}")

        # Count occurrences of each emotion
        emotion_count = {}
        for emotion in emotion_labels:
            if emotion in emotion_count:
                emotion_count[emotion] += 1
            else:
                emotion_count[emotion] = 1

        # Display emotion counts
        st.subheader("Emotion Counts:")
        for emotion, count in emotion_count.items():
            st.write(f"{emotion}: {count}")

        # Identify the most frequent emotion
        most_frequent_emotion = max(emotion_count, key=emotion_count.get)
        st.write(f"Most Frequent Emotion: {most_frequent_emotion}")

        # Check for and display related image
        image_base_name = most_frequent_emotion  # ใช้ชื่ออารมณ์เพื่อค้นหารูปภาพ
        image_name, image_data = get_image_from_db(image_base_name)
        if image_data:
            st.subheader("Related Image for Most Frequent Emotion:")
            st.image(BytesIO(image_data))

        # Display sunburst chart of emotion distribution
        st.subheader("Emotion Distribution Sunburst Chart:")
        st.markdown('<div class="sunburst-container">', unsafe_allow_html=True)
        sunburst_chart_fig = plot_sunburst_chart(emotion_labels)
        st.plotly_chart(sunburst_chart_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.header("Waveform")
            fig, ax = plt.subplots()
            librosa.display.waveshow(audio_data, sr=sr, color="purple", ax=ax)
            ax.set(xlabel='Time (s)', ylabel='Amplitude', title='Waveform')
            st.pyplot(fig)
            st.audio(audio_data, format='audio/wav', sample_rate=sample_rate)

        with col2:
            st.header("Spectrogram")
            fig, ax = plt.subplots()
            spec = librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max),
                                            sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='coolwarm')
            ax.set(title='Spectrogram')
            st.pyplot(fig)

        connection = create_connection()
        if connection:
            create_audio_table(connection)
            user_id = get_user_info(username)
            if user_id:
                try:
                    # Save recorded audio to database
                    insert_audio(connection, user_id, "recorded_audio", emotion_labels, most_frequent_emotion)
                    st.success("Recorded audio segments saved to MySQL database.")
                except Exception as e:
                    st.error(f"Error saving recorded audio: {e}")
            else:
                st.error("User ID not found.")
            connection.close()
            
if __name__ == "__main__":
    if 'username' not in st.session_state:
        st.session_state['username'] = 'default_user'  # Replace with actual user fetching mechanism
    deploy(st.session_state['username'])

