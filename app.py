from flask import Flask, request, render_template, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/f')
def my_form():
    return render_template('form.html')


@app.route('/f', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    
    #convert to lowercase
    text1 = request.form['text1'].lower()
    
    text_final = ''.join(c for c in text1 if not c.isdigit())
    
    #remove punctuations
    #text3 = ''.join(c for c in text2 if c not in punctuation)
        
    #remove stopwords    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)

    return render_template('form.html', final=compound, text1=text_final,text2=dd['pos'],text5=dd['neg'],text4=compound,text3=dd['neu'])




@app.route('/voice', methods=['POST', 'GET'])
def voice():
    if request.method == "POST":
        audio_file = request.files['audio']

        if audio_file.name == "":
            return render_template("voice_base.html")
        else:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(audio_file)
            with audioFile as source:
                data = recognizer.record(source)

            try:
                text = recognizer.recognize_google(data)
                print(text)
            except sr.UnknownValueError:
                print("Not recognized!")

        # print(audio_file)

    return render_template("voice_base.html")




@app.route('/file_voice', methods=['POST'])
def file_voice():
    if request.method == "POST":
        print("Form Data received")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == "":
            return redirect(request.url)

        try:
            if file:
                recognizer = sr.Recognizer()
                audioFile = sr.AudioFile(file)
                with audioFile as source:
                    data = recognizer.record(source)
                text = recognizer.recognize_google(data, key=None)
                print(text)
                # return text
            
                stop_words = stopwords.words('english')
        
                #convert to lowercase
                text1 = text.lower()
                
                text_final = ''.join(c for c in text1 if not c.isdigit())
                
                #remove punctuations
                #text3 = ''.join(c for c in text2 if c not in punctuation)
                    
                #remove stopwords    
                processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

                sa = SentimentIntensityAnalyzer()
                dd = sa.polarity_scores(text=processed_doc1)
                compound = round((1 + dd['compound'])/2, 2)

                return render_template('form.html', final=compound, text1=text_final,text2=dd['pos'],text5=dd['neg'],text4=compound,text3=dd['neu'])





        except Exception as e:
            print(e)

    return "ERROR"



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002, threaded=True)
