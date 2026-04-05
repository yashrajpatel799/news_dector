import numpy as np
import pickle
import requests
from newspaper import Article
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import os ,re
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient
import nltk
nltk.download('punkt')
load_dotenv() 
MONGO_URI = os.getenv("MONGO_URL")
client = MongoClient(MONGO_URI)
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


db = client["fake_news_db"] 
logs_collection = db["prediction_logs"]


trusted_source = [
  "thehindu.com",
  "indianexpress.com",
  "hindustantimes.com",
  "ndtv.com",
  "business-standard.com",
  "indiatoday.in",
  "thewire.in",
  "scroll.in",
  "timesofindia.indiatimes.com",
  "deccanherald.com",
  "telegraphindia.com",
  "livemint.com",
  "outlookindia.com",
  "firstpost.com",
  "news18.com",
  "caravanmagazine.in",
  "newslaundry.com",
  "freepressjournal.in",
  "dnaindia.com",
  "financialexpress.com",
  "bbc.com/news",
  "reuters.com",
  "apnews.com",
  "theguardian.com/international",
  "nytimes.com",
  "wsj.com",
  "aljazeera.com",
  "npr.org",
  "cnn.com",
  "ft.com",
  "economist.com",
  "dw.com/en",
  "france24.com/en",
  "cbc.ca/news",
  "abc.net.au/news",
  "japantimes.co.jp",
  "scmp.com"
]
def check_source(url):
    return any(source in url.lower() for source in trusted_source)
def predict_news(title,text):
    content = title + " "+ text
    content_vec =vectorizer.transform([content])
    length = len(content)
    title_vec =vectorizer.transform([title])
    text_vec = vectorizer.transform([text])
    similarity = cosine_similarity(title_vec,text_vec)[0][0]
    extra_features= np.array([[similarity,length]])
    final_input =hstack([content_vec,extra_features])
    prediction = model.predict(final_input)[0]
    prob =model.predict_proba(final_input)[0]
    confidence= float(max(prob)) *100
    return {
        "prediction" : "Fake" if prediction == 0 else "True",
        "confidence" :  round(confidence,2)
    } 
def verify_news(title):
    query = " ".join(title.split()[:8]) # Using 8 words for better specificity
    articles = []
    try:
        res1 = requests.get(f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}").json()
        articles.extend(res1.get("articles", []))
    except: pass
    try:
        res2 = requests.get(f"https://gnews.io/api/v4/search?q={query}&token={GNEWS_API_KEY}").json()
        articles.extend(res2.get("articles", []))
    except: pass
    if not articles:
        return {"verification_status": "Suspicious (No sources found)", "matched_articles": 0}
    api_titles = [a.get("title", "") for a in articles]
    all_titles = [title] + api_titles
    tfidf_matrix = vectorizer.transform(all_titles)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    strong_matches = sum(1 for score in similarities if score > 0.4)
    max_similarity = max(similarities) if len(similarities) > 0 else 0
    if max_similarity > 0.7 or strong_matches >= 2:
        status = "Highly Verified"
    elif max_similarity > 0.3 or strong_matches >= 1:
        status = "Likely True"
    else:
        status = "Suspicious (Topic found but content differs)"

    return {
        "verification_status": status,
        "matched_articles": strong_matches,
        "max_similarity_score": round(float(max_similarity), 2)
    }     
def extract_article(url):
    article=Article(url)
    article.download()
    article.parse()

    if not  article.text:
        raise Exception(" Could not extract article text")
    return {
        "title": article.title,
        "text" : article.text,
        "source" :article.source_url,
        "published_date" :str(article.publish_date)  if article.publish_date else "Not Available"
    }
client = genai.Client(api_key=GEMINI_API_KEY)

def predict_news_gemini(title=None, text=None, url=None):
    try:
        content = f"""
        Analyze the following news and respond STRICTLY in this format:
        AI_Verdict: True or Fake
        AI_Confidence: number (0-100)
        AI_Reason:  explanation
        Published_date:date of published news
        News:
        Title: {title if title else "N/A"}
        Text: {text if text else "N/A"}
        URL: {url if url else "N/A"}
        """
        response = client.models.generate_content(
            model="gemini-flash-latest",   # ✅ FIXED MODEL
            contents=content
        )
        res_text = response.text
        print(res_text)
        verdict = re.search(r"AI_Verdict:\s*(True|Fake)", res_text)
        conf = re.search(r"AI_Confidence:\s*(\d+)", res_text)
        reason = re.search(r"AI_Reason:\s*(.*)", res_text, re.DOTALL)
        published_date = re.search(r"Published_date:\s*(.*)", res_text)
        return {
            "verdict": verdict.group(1) if verdict else "Unverified",
            "confidence": conf.group(1) if conf else "0",
            "reason": reason.group(1) if reason else "No reason provided",
            "published_date": published_date.group(1) if published_date else "No reason provided"
        }
    except Exception as e:
        return {
            "verdict": "AI Error",
            "confidence": "0",
            "reason": str(e),
            "published_date": "N/A"
        }
def final(title, text, url=None):
    ml_result = predict_news(title, text) # Your original function
    verification = verify_news(title)
    gemini_result = predict_news_gemini(title, text, url)

    return {
        "model_prediction": ml_result["prediction"],
        "confidence": ml_result["confidence"],
        "fact_check_status": verification["verification_status"],
        "matched_sources": verification["matched_articles"],
        "is_trusted_source": check_source(url) if url else False,
        "Ai_verdict":gemini_result["verdict"],
        "Ai_confindance":gemini_result["confidence"],
        "Ai_reason":gemini_result["reason"],
        "published_Date":gemini_result["published_date"]
        
    }
def save_to_mongodb(log_data):
    try:
        # Prepare the document (This is your "row")
        document_to_save = {
            "timestamp": datetime.now(),
            "type": log_data["type"],
            "input_text": log_data["input"],
            "prediction": log_data["prediction"],
            "confidence": log_data["confidence"],
            "fact_check_status": log_data["verification"],
            "matched_articles": log_data["matched"],
            "ai_verdict": log_data.get("ai_verdict", "N/A"),
            "ai_reason": log_data.get("ai_reason", "N/A")
        }
        

        print("✅ Log successfully synced to MongoDB Atlas")
        result = logs_collection.insert_one(document_to_save)
        
        if result.inserted_id:
            print(f"✅ Log saved to MongoDB! (ID: {result.inserted_id})")
        return True
    except Exception as e:
        print(f"❌ MongoDB Logging Error: {e}")

