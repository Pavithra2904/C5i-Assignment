import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests
from google.cloud import speech
from flask import Flask, render_template, request, jsonify
import os


nltk.download('stopwords')
nltk.download('vader_lexicon')  # Download VADER lexicon for sentiment analysis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg', 'mp4', 'mov', 'avi', 'mkv'}

API_URL_SUMMARIZATION = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
API_URL_QA = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": "Bearer YOUR_HUGGINGFACE_API_TOKEN"}

def summarize(text):
    response = requests.post(API_URL_SUMMARIZATION, headers=headers, json={"inputs": text})
    return response.json()[0]['summary_text']

def answer_question(question, context):
    response = requests.post(API_URL_QA, headers=headers, json={
        "inputs": {
            "question": question,
            "context": context
        }
    })
    return response.json()['answer']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    video.close()

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)
    with audio_file as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Speech Recognition could not understand audio"
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service"

def extract_topics(text, num_topics=3):
    stop_words = set(stopwords.words('english'))
    stop_words = list(stop_words)

    vectorizer = CountVectorizer(stop_words=stop_words)
    doc_term_matrix = vectorizer.fit_transform([text])

    tokens = [word for word in text.lower().split() if word not in stop_words]
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics()
    return topics

def extract_metadata(text):
    word_count = len(text.split())
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.lower().split() if word not in stop_words]

    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1

    return {
        'word_count': word_count,
        'most_common_words': sorted(frequency.items(), key=lambda item: item[1], reverse=True)[:10]
    }

def sentiment_distribution(text):
    sia = SentimentIntensityAnalyzer()
    sentences = text.split('.')
    positive, negative, neutral = 0, 0, 0

    for sentence in sentences:
        sentiment = sia.polarity_scores(sentence)
        if sentiment['compound'] >= 0.05:
            positive += 1
        elif sentiment['compound'] <= -0.05:
            negative += 1
        else:
            neutral += 1
    
    total = positive + negative + neutral
    return {
        'positive': positive / total * 100 if total > 0 else 0,
        'negative': negative / total * 100 if total > 0 else 0,
        'neutral': neutral / total * 100 if total > 0 else 0
    }

def generate_insights(text):
    metadata = extract_metadata(text)
    sentiment = sentiment_distribution(text)

    most_common_words = metadata['most_common_words']
    positive_words = [word for word, count in most_common_words if sentiment['positive'] > sentiment['negative']]
    negative_words = [word for word, count in most_common_words if sentiment['negative'] > sentiment['positive']]
    
    return {
        'key_takeaways': f"Most common positive words: {positive_words}, Most common negative words: {negative_words}",
        'controversial_points': f"Mixed sentiment detected in common words: {positive_words + negative_words}",
        'consensus': f"Sentiment distribution shows {sentiment['positive']}% positive, {sentiment['negative']}% negative, and {sentiment['neutral']}% neutral."
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the file
        audio_path = 'extracted_audio.wav'
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            extract_audio_from_video(file_path, audio_path)
        else:
            audio_path = file_path

        # Transcribe the audio
        text = transcribe_audio(audio_path)

        # Topic Extraction
        topics = extract_topics(text)

        # Metadata Extraction
        metadata = extract_metadata(text)

        # Sentiment Analysis
        sentiment = sentiment_distribution(text)

        # Insight Generation
        insights = generate_insights(text)

        # Clean up
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            os.remove(audio_path)
        os.remove(file_path)

        # Render the result page with the extracted information
        return render_template('results.html', transcription=text, topics=topics, metadata=metadata, sentiment=sentiment, insights=insights)

    return jsonify({'error': 'Invalid file type'}), 400



if __name__ == '__main__':
    app.run(debug=True)
