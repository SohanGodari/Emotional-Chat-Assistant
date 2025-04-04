import os
import re
import json
import time
import joblib
import psycopg2
import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import matplotlib.pyplot as plt
import librosa
import random
import nltk
import contractions
from datetime import datetime, timedelta
from gtts import gTTS
from deep_translator import GoogleTranslator
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model  # type:ignore
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Optional, Literal, Any, Dict, List, Tuple
st.set_page_config(layout="wide", page_title="Emotional AI Support", page_icon="💡", initial_sidebar_state="expanded")

@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-mpnet-base-v2')

embedder = get_embedder()

@st.cache_resource
def load_keras_model(path: str):
    return load_model(path)

model = load_keras_model(r'Models/model_emotion.h5')
model_text = load_keras_model(r'Models/emotion_text.h5')

def preprocess_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = ' '.join(tokens)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text

Emotion = Literal["anger", "boredom", "disgust", "anxiety/fear", "happiness", "sadness", "neutral"]

class RLAgent:
    def __init__(
        self,
        action_modifiers: List[int],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0.01,
        q_table: Optional[Dict[Tuple[Any, int], float]] = None
    ) -> None:
        self.actions: List[int] = action_modifiers
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.min_epsilon: float = min_epsilon
        self.q_table: Dict[Tuple[Any, int], float] = q_table if q_table is not None else {}
        # For UCB exploration: track the number of times each (emotion, action) pair is selected.
        self.sa_counts: Dict[Tuple[Any, int], int] = {}
        # Base UCB constant (tunable parameter)
        self.c_ucb: float = 2.0

    def get_q(self, emotion: str, action: int) -> float:
        return self.q_table.get((emotion, action), 0.0)

    def choose_action(self, emotion: str) -> int:
        """
        Choose an action using a dynamic UCB exploration strategy.
        """
        # Use the current emotion as the state.
        state = emotion
        # Total count for all actions in this state; add 1 to avoid log(0)
        total_count = sum(self.sa_counts.get((state, a), 0) for a in self.actions) + 1

        # Compute a dynamic UCB constant that decays as more data is gathered.
        dynamic_c = self.c_ucb / np.sqrt(total_count)

        ucb_values = []
        for a in self.actions:
            count = self.sa_counts.get((state, a), 0)
            bonus = dynamic_c * np.sqrt(np.log(total_count) / (count + 1))
            ucb_value = self.get_q(emotion, a) + bonus
            ucb_values.append(ucb_value)
        
        max_ucb = max(ucb_values)
        # In case of ties, select randomly among the best actions.
        best_actions = [a for a, val in zip(self.actions, ucb_values) if val == max_ucb]
        chosen_action = random.choice(best_actions)
        # Update count for the chosen state-action pair.
        self.sa_counts[(state, chosen_action)] = self.sa_counts.get((state, chosen_action), 0) + 1
        return chosen_action

    def update(self, emotion: str, action: int, reward: float, next_emotion: Optional[str] = None) -> None:
        old_q = self.get_q(emotion, action)
        if next_emotion is not None:
            future_reward = max([self.get_q(next_emotion, a) for a in self.actions])
            target = reward + self.gamma * future_reward
        else:
            target = reward
        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[(emotion, action)] = new_q
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, user_email: str) -> None:
        user_dir = os.path.join("data", user_email)
        os.makedirs(user_dir, exist_ok=True)
        joblib.dump(self.q_table, os.path.join(user_dir, "rl_agent.pkl"))

    @classmethod
    def load(cls, user_email: str, action_modifiers: List[int],
             alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
             epsilon_decay: float = 0.99, min_epsilon: float = 0.01) -> "RLAgent":
        user_dir = os.path.join("data", user_email)
        filepath = os.path.join(user_dir, "rl_agent.pkl")
        if os.path.exists(filepath):
            q_table = joblib.load(filepath)
            return cls(action_modifiers, alpha, gamma, epsilon, epsilon_decay, min_epsilon, q_table)
        else:
            return cls(action_modifiers, alpha, gamma, epsilon, epsilon_decay, min_epsilon)

def get_user_rl_agent(user_email: str) -> RLAgent:
    return RLAgent.load(user_email, action_modifiers=[0, 1, 2, 3, 4])

action_modifiers = {
    0: "Please respond in a supportive and factual manner.",
    1: "Please respond with empathy and understanding, considering the user's emotional state.",
    2: "Please provide a creative, supportive, and imaginative response that is both insightful and engaging.",
    3: "Please respond with a touch of humor while remaining supportive.",
    4: "Please provide a concise and direct response without unnecessary elaboration."
}

if "logged_in" in st.session_state and st.session_state["logged_in"]:
    if "user_info" in st.session_state and "rl_agent" not in st.session_state:
        user_email = st.session_state["user_info"]["email"]
        st.session_state.rl_agent = get_user_rl_agent(user_email)
# pkl_file = "data/test@gmail.com/rl_agent.pkl"

# # Load the data
# q_table = joblib.load(pkl_file)

# print("Loaded Q-table:", q_table)
DB_PARAMS = {
    "dbname": "knowledge_base",
    "user": "postgres",
    "password": "root",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    try:
        return psycopg2.connect(**DB_PARAMS)
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

def initialize_global_kb():
    print("Initializing knowledge base...")
    conn = get_db_connection()
    if conn is None:
        return
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id SERIAL PRIMARY KEY,
            query TEXT UNIQUE NOT NULL,
            response TEXT NOT NULL,
            emotion TEXT,
            embedding TEXT,
            frequency INT DEFAULT 1,
            last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Knowledge base initialized.")

def lookup_global_kb(query):
    conn = get_db_connection()
    if conn is None:
        return None
    cur = conn.cursor()

    normalized_query = preprocess_text(query)
    cur.execute("SELECT response FROM knowledge_base WHERE query = %s;", (normalized_query,))
    result = cur.fetchone()
    if result:
        cur.execute("UPDATE knowledge_base SET frequency = frequency + 1, last_used = %s WHERE query = %s;",
                    (datetime.now(), normalized_query))
        conn.commit()
        cur.close()
        conn.close()
        return result[0]

    cur.execute("SELECT query, response, embedding FROM knowledge_base;")
    rows = cur.fetchall()
    if not rows:
        cur.close()
        conn.close()
        return None

    input_embedding = embedder.encode(normalized_query).astype(np.float32)

    best_score = 0
    best_response = None
    best_db_query = None

    for db_query, db_response, embedding_str in rows:
        try:
            db_embedding = np.array(json.loads(embedding_str), dtype=np.float32)
            similarity = util.cos_sim(input_embedding, db_embedding).item()
            # print(f"[DEBUG] Similarity for query '{db_query}': {similarity*100:.2f}%")
            if similarity > best_score:
                best_score = similarity
                best_response = db_response
                best_db_query = db_query
        except Exception as e:
            print(f"Error processing embedding for query '{db_query}': {e}")

    SIMILARITY_THRESHOLD = 0.75
    if best_score > SIMILARITY_THRESHOLD:
        cur.execute("UPDATE knowledge_base SET frequency = frequency + 1, last_used = %s WHERE query = %s;",
                    (datetime.now(), best_db_query))
        conn.commit()
        cur.close()
        conn.close()
        return best_response

    cur.close()
    conn.close()
    return None

def update_global_kb(original_query, response, emotion=None):
    """
    1. Keep `original_query` exactly as the user typed it.
    2. For embeddings, you might normalize or translate to English, but do NOT overwrite `original_query`.
    """
    conn = get_db_connection()
    if conn is None:
        print("Database connection failed!")
        return

    # 1) Save the raw user query for the DB record
    raw_user_query = original_query.strip()

    # 2) Build the normalized/translated text for embedding
    #    e.g. if you're translating to English for semantic search:
    #    english_query = translate(original_query, detect_lang(original_query), "en")
    #    Or if you only do basic normalization:
    normalized_query = preprocess_text(original_query)

    new_embedding = embedder.encode(normalized_query).astype(np.float32)

    # Debug print
    # print(f"[DEBUG] Original query: '{original_query}'")
    # print(f"[DEBUG] Normalized query: '{normalized_query}'")

    cur = conn.cursor()
    # Retrieve all existing rows
    cur.execute("SELECT query, response, embedding FROM knowledge_base;")
    rows = cur.fetchall()

    best_score = 0
    best_query = None
    for db_query, db_response, embedding_str in rows:
        try:
            db_embedding = np.array(json.loads(embedding_str), dtype=np.float32)
            similarity = util.cos_sim(new_embedding, db_embedding).item()
            # print(f"[DEBUG] Similarity with '{db_query}': {similarity:.3f}")
            if similarity > best_score:
                best_score = similarity
                best_query = db_query
        except Exception as e:
            print(f"[DEBUG] Error processing embedding for query '{db_query}': {e}")

    # print(f"[DEBUG] Best similarity found: {best_score:.3f} for query '{best_query}'")

    SIMILARITY_THRESHOLD = 0.75
    if best_score > SIMILARITY_THRESHOLD and best_query is not None:
        # print(f"[DEBUG] Updating existing entry for '{best_query}'")
        cur.execute("""
            UPDATE knowledge_base
            SET response = %s,
                emotion = %s,
                frequency = frequency + 1,
                last_used = %s
            WHERE query = %s;
        """, (response, emotion, datetime.now(), best_query))
        conn.commit()
        cur.close()
        conn.close()
        return

    # If no high similarity found, we insert a new row
    # print(f"[DEBUG] Inserting new query: '{raw_user_query}'")
    embedding_str = json.dumps(new_embedding.tolist())
    # Storing original vs. normalized
    cur.execute("""
        INSERT INTO knowledge_base (query, response, emotion, embedding)
        VALUES (%s, %s, %s, %s);
    """, (raw_user_query,  response, emotion, embedding_str))

    conn.commit()
    cur.close()
    conn.close()
    # print("[DEBUG] Query successfully updated in DB.")

def prune_old_queries(threshold_days=30):
    cutoff_time = datetime.now() - timedelta(days=threshold_days)
    conn = get_db_connection()
    if conn is None:
        print("Database connection error.")
        return

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM knowledge_base WHERE last_used < %s;", (cutoff_time,))
    pruned_count = cur.fetchone()[0]

    if pruned_count > 0:
        cur.execute("DELETE FROM knowledge_base WHERE last_used < %s;", (cutoff_time,))
        conn.commit()
    cur.close()
    conn.close()

session_state = st.session_state
if "user_index" not in session_state:
    session_state["user_index"] = 0


genai.configure(api_key="AIzaSyArFsF8XTEyuPDbQhtvGjZfygziLN6RF7o")
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}
@st.cache_resource
def get_gen_model():
    return genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

gen_model = get_gen_model()

LOGIN_FILE = "logged_in_users.json"

def save_login(email):
    login_data = {"logged_in_user": email}
    with open(LOGIN_FILE, "w") as file:
        json.dump(login_data, file)

def load_login():
    if os.path.exists(LOGIN_FILE):
        try:
            with open(LOGIN_FILE, "r") as file:
                login_data = json.load(file)
                return login_data.get("logged_in_user")
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    return None

def create_account(name, email, age, sex, password, emotions, json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "emotions": emotions,
            "chats": []
        }
        data["users"].append(user_info)
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None

def signup(json_file_path="data.json"):
    st.title("Signup")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")
        emotions = []
        if st.form_submit_button("Signup"):
            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
            if not re.match(email_pattern, email):
                st.error("Invalid email address. Please enter a valid email.")
                return
            password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{6,}$'
            if not re.match(password_pattern, password):
                st.error("Password must be at least 6 characters long and include at least one number, one lowercase letter, and one uppercase letter.")
                return
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, emotions, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.session_state.chat_id = None
                st.session_state.page = "AI assistant"
                st.rerun()
            else:
                st.error("Passwords do not match. Please try again.")

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                save_login(username)
                st.success("Login successful!")
                return user
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None

def login(json_file_path="data.json"):
    st.title("Login")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")
    login_button = st.button("Login")
    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
            save_login(username)
            st.session_state.chat_id = None
            st.session_state.page = "AI assistant"
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

def initialize_database(json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None

def predict_emotion_text(text):
    prompt = f"""Respond in a single word. Identify the emotion detected in the given text. 
Strictly choose only from the provided emotions. 
Make sure to use exact same spelling and case like capital and small.
emotions: ['anger','boredom','disgust','anxiety/fear','happiness','sadness','neutral']
text: {text}"""
    response = gen_model.generate_content(prompt)
    predicted_emotion = response.text.strip()
    if predicted_emotion.lower().startswith("neutral"):
        predicted_emotion = "neutral"
    return {"emotion": predicted_emotion}

def emotion_rec(text: str) -> str:
    try:
        emotion = predict_emotion_text(text)
        return emotion['emotion']
    except Exception as e:
        print(f"Error in emotion recognition: {e}")
        return "Neutral"

def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result

def predict_emotion(file_path, model=model):
    import librosa
    normalized_path = os.path.normpath(file_path)
    data, sample_rate = librosa.load(normalized_path, duration=2.5, offset=0.6)
    features = extract_features(data, sample_rate)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)
    prediction = model.predict(features)
    predicted_class_index = np.argmax(prediction)
    emotion_dict = {
        0: 'anger',
        1: 'boredom',
        2: 'disgust',
        3: 'anxiety/fear',
        4: 'happiness',
        5: 'sadness',
        6: 'neutral'
    }
    predicted_emotion = emotion_dict[predicted_class_index]
    return predicted_emotion

MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '🌟'

def apply_accessibility_styles(font_size: int, high_contrast: bool):
    base_styles = f"""
        <style>
        .stTextInput, .stMarkdown, .stSelectbox, .stButton > button {{
            font-size: {font_size}px !important;
            line-height: 1;
        }}
        .element-container {{
            margin-bottom: 0.5em;
        }}
        *:focus {{
            outline: 3px solid #4CAF50 !important;
            outline-offset: 2px !important;
        }}
        </style>
    """
    if high_contrast:
        base_styles += """
        <style>
        /* High contrast overrides */
        .main, .main .block-container, [data-testid="stAppViewContainer"],
        section[data-testid="stSidebar"], div.stApp,
        .streamlit-wide, [data-testid="stHeader"],
        footer, .reportview-container, header,
        [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],[data-testid="stBottomBlockContainer"], .footer-text {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stChatInput"], .stChatFloatingInputContainer, div[data-testid="stChatInputContainer"] {
            background-color: #000000 !important;
            border-color: #666666 !important;
        }
        [data-testid="chatInputCard"] {
            background-color: #333333 !important;
            border: 1px solid #666666 !important;
        }
        [data-testid="stChatInput"] textarea, .stChatInputContainer textarea {
            background-color: #333333 !important;
            color: #FFFFFF !important;
            border-color: #666666 !important;
        }
        [data-testid="stChatInput"] button, .stChatInputContainer button {
            background-color: #333333 !important;
            color: #FFFFFF !important;
            border-color: #666666 !important;
        }
        iframe { background-color: #000000 !important; }
        div[data-testid="stDecoration"] { background-color: #000000 !important; display: none; }
        div[data-testid="stToolbar"] { background-color: #000000 !important; right: 0; top: 0; }
        .stDeployButton { background-color: #000000 !important; color: #FFFFFF !important; }
        body, #root > div:first-child { background-color: #000000 !important; }
        [data-testid="stStatusWidget"] { background-color: #000000 !important; color: #FFFFFF !important; }
        div[data-testid="stMarkdownContainer"], div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] span {
            color: #FFFFFF !important;
        }
        .stChatMessage, div[data-testid="stChatMessageContent"] {
            background-color: #333333 !important;
            border: 0px solid #666666 !important;
            color: #FFFFFF !important;
        }
        .stChatMessageAvatar { background-color: #333333 !important; }
        .stChatInputContainer, textarea[aria-label="Type your message here... (Press Enter to send)"], div[data-testid="stChatInput"] {
            background-color: #333333 !important;
            color: #FFFFFF !important;
            border-color: #666666 !important;
        }
        .stTextInput > div > div > input, .stSelectbox > div > div, .stTextArea textarea {
            background-color: #333333 !important;
            color: #FFFFFF !important;
            border: 1px solid #666666 !important;
        }
        .stButton > button {
            background-color: #666666 !important;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF !important;
        }
        .stSlider > div > div > div { background-color: #666666 !important; }
        h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }
        div[data-baseweb="tooltip"], div[data-baseweb="popover"] {
            background-color: #333333 !important;
            color: #FFFFFF !important;
            border: 1px solid #666666 !important;
        }
        </style>
        """
    st.markdown(base_styles, unsafe_allow_html=True)

def save_emotion_to_json(emotion, current_user_email, json_file_path="data.json"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    new_entry = {"emotion": emotion, "timestamp": timestamp}
    with open(json_file_path, "r") as file:
        data = json.load(file)
    for user in data["users"]:
        if user["email"] == current_user_email:
            if "emotions" not in user:
                user["emotions"] = []
            user["emotions"].append(new_entry)
            break
    with open(json_file_path, "w") as file:
        json.dump(data, file, indent=4)

def set_page_config():
    st.set_page_config(
        page_title="Emotional AI Assistant",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
        .stAppHeader { background-color: transparent !important; padding: 1rem; }
        .app-title { font-size: 2.5rem; font-weight: bold; text-align: center; margin: 0; padding: 1rem; }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<div class="stAppHeader"><h1 class="app-title">🌟 AI Mental Health Companion</h1></div>', unsafe_allow_html=True)

def setup_assistant_prompt():
    return """You are an advanced AI assistant (Gemini).
    1. If the user asks a general or factual question, respond with the best possible, concise answer.
    2. If the user indicates they need emotional support or the recognized emotion is negative 
       (anger, sadness, anxiety/fear, boredom, happiness, neutral, disgust), respond empathetically.
       - Use supportive language and a calm tone,
       - But avoid giving direct medical or mental health advice.
       - Acknowledge the user’s feelings.
    3. Always factor in the user's recognized emotion:
       - If the emotion is positive (happiness, etc.), maintain an upbeat tone.
       - If negative, respond gently and with empathy.
    4. Use simple, clear language.
    5. Do not provide disclaimers like "I'm not a professional." Just be supportive and understanding.
    6. Keep your responses short and to the point.
    """

os.makedirs('data/', exist_ok=True)
languages = {
    "English": "en",
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Chichewa": "ny",
    "Chinese (simplified)": "zh-cn",
    "Chinese (traditional)": "zh-tw",
    "Corsican": "co",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Esperanto": "eo",
    "Estonian": "et",
    "Filipino": "tl",
    "Finnish": "fi",
    "French": "fr",
    "Frisian": "fy",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian creole": "ht",
    "Hausa": "ha",
    "Hawaiian": "haw",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hmong": "hmn",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Igbo": "ig",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jw",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Khmer": "km",
    "Korean": "ko",
    "Kurdish (kurmanji)": "ku",
    "Kyrgyz": "ky",
    "Lao": "lo",
    "Latin": "la",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Myanmar (burmese)": "my",
    "Nepali": "ne",
    "Norwegian": "no",
    "Odia": "or",
    "Pashto": "ps",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Samoan": "sm",
    "Scots gaelic": "gd",
    "Serbian": "sr",
    "Sesotho": "st",
    "Shona": "sn",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tajik": "tg",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uyghur": "ug",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Zulu": "zu"
}

genai.configure(api_key="AIzaSyArFsF8XTEyuPDbQhtvGjZfygziLN6RF7o")
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}
gen_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

def create_sidebar():
    with st.sidebar:
        if st.session_state.get("logged_in"):
            if st.button("Logout", help="Click to logout"):
                logout()
        st.markdown("## AI Mental Health Companion")
        st.warning("## Settings ")
        st.markdown("## Accessibility Options")
        st.markdown("### Text Size")
        font_size = st.slider("Select text size", min_value=14, max_value=32, value=18, step=2, help="Adjust text size")
        st.markdown("### Display Settings")
        high_contrast = st.toggle("High Contrast Mode", value=False, help="Enable for better visibility")
        st.markdown("## Previous Conversations")
        return_to_chats(font_size, high_contrast)
        return font_size, high_contrast

def generate_response_(text):
    prompt = f"""You are a helpful chatbot for mental health support. Respond to the given text accordingly and empathetically.
        Make sure to give responses and not include headers or names.
        \n\ntext:{text}"""
    try:
        response = gen_model.generate_content(prompt)
        if len(response.text) == 0:
            return "I'm sorry, I don't understand. Can you please rephrase?"
        return response.text
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I'm sorry, couldn't generate a response"

def return_to_chats(font_size: int, high_contrast: bool):
    if 'chat_title' not in st.session_state:
        st.session_state.chat_title = "New Chat"
    current_user_email = session_state["user_info"]["email"]
    user_dir = f'data/{current_user_email}'
    os.makedirs(user_dir, exist_ok=True)
    with open("data.json", "r") as file:
        data = json.load(file)
    user_data = next((u for u in data["users"] if u["email"] == current_user_email), None)
    user_chats = user_data.get("chats", []) if user_data else []
    valid_chats = [chat for chat in user_chats if (time.time() - chat["created_at"]) <= 30 * 86400]
    chat_options = [chat["id"] for chat in valid_chats]
    chat_labels = {chat["id"]: chat["name"] for chat in valid_chats}
    new_chat_id = f'{current_user_email}-{time.time()}'
    st.markdown("### Start a New Chat")
    with st.form(key='chat_naming_form'):
        custom_title = st.text_input("Name your chat", value=f"Chat-{datetime.now().strftime('%Y-%m-%d %H:%M')}", key='new_chat_name')
        submitted = st.form_submit_button('Create')
        if submitted:
            existing_names_lower = [chat["name"].lower() for chat in valid_chats]
            if custom_title.strip().lower() in existing_names_lower:
                st.error("Chat name already exists. Please use a unique name.")
            else:
                new_chat = {"id": new_chat_id, "name": custom_title, "created_at": time.time()}
                user_data["chats"].append(new_chat)
                with open("data.json", "w") as f:
                    json.dump(data, f, indent=4)
                st.session_state.chat_id = new_chat_id
                st.session_state.chat_title = custom_title
                joblib.dump([], f'{user_dir}/{new_chat_id}-st_messages')
                joblib.dump([], f'{user_dir}/{new_chat_id}-gemini_messages')
                st.rerun()
    st.markdown("### Previous Conversations")
    all_options = ["No chat selected"] + chat_options
    default_index = all_options.index(st.session_state.chat_id) if st.session_state.chat_id in chat_options else 0
    selected_chat = st.selectbox(
        "Select a conversation",
        all_options,
        format_func=lambda x: "No chat selected" if x == "No chat selected" else chat_labels.get(x, "Unnamed Chat"),
        index=default_index
    )
    if selected_chat == "No chat selected":
        if st.session_state.chat_id is not None:
            st.session_state.chat_id = None
            st.session_state.chat_title = "New Chat"
            st.session_state.messages = []
            st.session_state.gemini_history = []
            st.rerun()
    else:
        if selected_chat != st.session_state.get("chat_id"):
            st.session_state.chat_id = selected_chat
            st.session_state.chat_title = chat_labels.get(selected_chat, "New Chat")
            st.rerun()

def initialize_chat():
    current_user_email = session_state["user_info"]["email"]
    user_dir = f'data/{current_user_email}'
    os.makedirs(user_dir, exist_ok=True)
    try:
        st.session_state.messages = joblib.load(f'{user_dir}/{st.session_state.chat_id}-st_messages')
        st.session_state.gemini_history = joblib.load(f'{user_dir}/{st.session_state.chat_id}-gemini_messages')
    except:
        st.session_state.messages = []
        st.session_state.gemini_history = []
    generation_config = {"temperature": 0.5, "top_p": 0.95, "top_k": 64, "max_output_tokens": 1000}
    st.session_state.model = genai.GenerativeModel(model_name='gemini-1.5-flash', generation_config=generation_config)
    st.session_state.chat = st.session_state.model.start_chat(history=st.session_state.gemini_history)
    if not st.session_state.gemini_history:
        st.session_state.chat.send_message(setup_assistant_prompt())

def cleanup_expired_chats():
    current_user_email = session_state["user_info"]["email"]
    user_dir = f'data/{current_user_email}'
    with open("data.json", "r+") as file:
        data = json.load(file)
        for user in data["users"]:
            if user["email"] == current_user_email:
                valid_chats = []
                for chat in user.get("chats", []):
                    if (time.time() - chat["created_at"]) <= 30 * 86400:
                        valid_chats.append(chat)
                    else:
                        for suffix in ["-st_messages", "-gemini_messages"]:
                            file_path = f'{user_dir}/{chat["id"]}{suffix}'
                            if os.path.exists(file_path):
                                os.remove(file_path)
                user["chats"] = valid_chats
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

def display_message_with_emotion(message: dict, font_size: int):
    with st.chat_message(name=message['role'], avatar=message.get('avatar')):
        content = message['content']
        if 'emotion' in message:
            st.markdown(f"<p style='font-size: {font_size}px; color: #666;'><em>Emotion: {message['emotion']}</em></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: {font_size}px;'>{content}</p>", unsafe_allow_html=True)

def display_messages(font_size: int):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("emotion"):
                st.markdown(f"<p style='font-size: {font_size}px; color: #666;'><em>Emotion: {msg['emotion']}</em></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: {font_size}px;'>{msg['content']}</p>", unsafe_allow_html=True)

def load_user_emotions(user_email, json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    user_data = next((user for user in data['users'] if user['email'] == user_email), None)
    return user_data['emotions'] if user_data and "emotions" in user_data else []

def logout():
    preserved_state = {'page': 'Home', 'chat_id': None, 'initialized': st.session_state.get('initialized', False)}
    st.session_state.clear()
    st.session_state.update(preserved_state)
    if os.path.exists(LOGIN_FILE):
        os.remove(LOGIN_FILE)
    st.rerun()

# ------------------- NEW: Build Full Conversation Context -------------------
def build_full_context():
    """
    Builds a single text string containing the entire conversation so far.
    """
    conversation_str = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            conversation_str += f"User: {msg['content']}\n"
        else:
            conversation_str += f"AI: {msg['content']}\n"
    return conversation_str

# ------------------- Updated Personalized Response Function -------------------
def generate_personalized_response(query, recognized_emotion, user_email, action_modifier):
    """
    Includes the entire conversation (so far) plus system instructions and user emotion.
    This ensures the model can reference previous messages in the same chat.
    """
    system_instructions = setup_assistant_prompt()
    emotion_summary = build_emotion_summary(user_email)
    
    # Build conversation context from st.session_state.messages
    conversation_history = build_full_context()

    final_prompt = f"""
{system_instructions}

{emotion_summary}

Here is the conversation so far:
{conversation_history}

The user just said: "{query}"
Recognized Emotion: "{recognized_emotion}"

{action_modifier}

Now, please provide a personalized, empathetic, and context-aware response:
"""

    try:
        response = gen_model.generate_content(final_prompt)
        generated = response.text.strip()
        if not generated:
            return "I'm sorry, I couldn't generate a response. Please try again."
        return generated
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm sorry, couldn't generate a response."

def build_emotion_summary(user_email, json_file_path="data.json"):
    user = get_user_info(user_email, json_file_path)
    if not user or "emotions" not in user or not user["emotions"]:
        return "No recorded emotions."
    emotions = user["emotions"]
    emotion_counts = {}
    for entry in emotions:
        em = entry.get("emotion", "neutral")
        emotion_counts[em] = emotion_counts.get(em, 0) + 1
    summary = ", ".join([f"{k}: {v}" for k, v in emotion_counts.items()])
    return f"User Emotion Summary: {summary}"

# ------------------- Text Assistant Function with Memory -------------------
def ai_assistant():
    if st.session_state.get("logged_in"):
        rl_agent = st.session_state.rl_agent
        local_action_modifiers = action_modifiers

        font_size, high_contrast = create_sidebar()
        apply_accessibility_styles(font_size, high_contrast)

        st.markdown(f"""
            <p style='font-size: {font_size}px;'>
                This AI assistant is here to help you! It's designed to be:
                <ul>
                    <li>Easy to read and understand</li>
                    <li>Patient and supportive</li>
                    <li>Flexible to your needs</li>
                    <li>Emotion-aware</li>
                </ul>
                Just type your question or request below.
            </p>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .stButton > button {
            border: none !important;
            box-shadow: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

        if st.session_state.chat_id is not None:
            # Load/initialize the chat
            initialize_chat()
            display_messages(font_size)

            user_input = st.chat_input('Type your message here... (Press Enter to send)',
                                       key="chat_input", max_chars=1000)
            if user_input:
                current_user_email = st.session_state["user_info"]["email"]
                detected_emotion = emotion_rec(user_input)
                save_emotion_to_json(detected_emotion, current_user_email)

                # Store user's message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "emotion": detected_emotion
                })

                # Check KB or generate a new response
                stored_response = lookup_global_kb(user_input)
                if stored_response:
                    final_response = stored_response
                else:
                    chosen_action = rl_agent.choose_action(detected_emotion)
                    st.session_state.last_action = chosen_action
                    modifier = local_action_modifiers[chosen_action]
                    final_response = generate_personalized_response(
                        user_input, detected_emotion, current_user_email, modifier
                    )
                    update_global_kb(user_input, final_response, emotion=detected_emotion)

                # Append AI's response
                st.session_state.messages.append({
                    "role": MODEL_ROLE,
                    "content": final_response,
                    "avatar": AI_AVATAR_ICON
                })
                st.session_state.last_ai_index = len(st.session_state.messages) - 1

                # Save to disk
                user_dir = f"data/{current_user_email}"
                os.makedirs(user_dir, exist_ok=True)
                joblib.dump(st.session_state.messages, f"{user_dir}/{st.session_state.chat_id}-st_messages")
                joblib.dump(st.session_state.gemini_history, f"{user_dir}/{st.session_state.chat_id}-gemini_messages")

                st.rerun()

            # Handle feedback on last AI message
            if "last_ai_index" in st.session_state and st.session_state.last_ai_index is not None:
                if st.session_state.last_ai_index < len(st.session_state.messages):
                    last_msg = st.session_state.messages[st.session_state.last_ai_index]
                    if last_msg["role"] == MODEL_ROLE:
                        col1, col2 = st.columns(2)
                        if col1.button("👍"):
                            # Positive feedback
                            # We still read the AI message's emotion here, but you could also
                            # retrieve the user emotion if you prefer to train on user emotion for "like" as well.
                            emotion = last_msg.get("emotion") or "neutral"
                            action = st.session_state.get("last_action", 0)
                            rl_agent.update(emotion, action, 1)
                            rl_agent.save(st.session_state["user_info"]["email"])
                            st.success("Thank you for your feedback!")
                            st.session_state.last_ai_index = None
                            st.rerun()

                        if col2.button("👎", key=f"dislike_feedback_{st.session_state.last_ai_index}"):
                            st.error("Feedback noted. Regenerating response...")

                            # Remove disliked AI response
                            st.session_state.messages.pop(st.session_state.last_ai_index)

                            # Find the user's last message to retrieve the *user's* emotion
                            user_msg = None
                            for i in reversed(range(len(st.session_state.messages))):
                                if st.session_state.messages[i]["role"] == "user":
                                    user_msg = st.session_state.messages[i]
                                    break

                            if not user_msg:
                                st.warning("No user message found to regenerate from.")
                                st.session_state.last_ai_index = None
                                st.rerun()

                            # Use the user's emotion for the Q-update
                            user_emotion = user_msg.get("emotion", "neutral")
                            action = st.session_state.get("last_action", 0)
                            rl_agent.update(user_emotion, action, -1)  # Negative feedback
                            rl_agent.save(st.session_state["user_info"]["email"])

                            # Generate new response
                            new_action = rl_agent.choose_action(user_emotion)
                            st.session_state.last_action = new_action
                            new_modifier = local_action_modifiers[new_action]
                            new_response = generate_personalized_response(
                                user_msg["content"], user_emotion,
                                st.session_state["user_info"]["email"],
                                new_modifier
                            )

                            # Append the new AI message
                            st.session_state.messages.append({
                                "role": MODEL_ROLE,
                                "content": new_response,
                                "avatar": AI_AVATAR_ICON
                            })
                            st.session_state.last_ai_index = len(st.session_state.messages) - 1

                            # Save again
                            user_dir = f"data/{st.session_state['user_info']['email']}"
                            os.makedirs(user_dir, exist_ok=True)
                            joblib.dump(st.session_state.messages, f"{user_dir}/{st.session_state.chat_id}-st_messages")
                            joblib.dump(st.session_state.gemini_history, f"{user_dir}/{st.session_state.chat_id}-gemini_messages")
                            st.rerun()

        else:
            st.info("Please create or select a conversation from the sidebar.")
    else:
        st.warning("Please login/signup to view the dashboard.")


# -------------- Voice Assistant and other code remains the same --------------

def play(text, lang):
    try:
        speech = gTTS(text=text, lang=lang, slow=False)
        speech.save('audio.mp3')
        if os.path.exists('audio.mp3'):
            os.remove('audio.mp3')
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def translate(text, source_language, target_language):
    texts = []
    while len(text) > 5000:
        index = 5000
        while text[index] != "." and index > 0:
            index -= 1
        if index == 0:
            index = 5000
        texts.append(text[:index])
        text = text[index:]
    texts.append(text)
    translated_text = ""
    for t in texts:
        translated_text += (
            GoogleTranslator(source=source_language, target=target_language).translate(t)
            + " "
        )
    return translated_text

def voice_assistant():
    """
    A voice-based assistant that uses the full conversation history so that the AI
    remembers previous interactions. It listens to the user, transcribes the speech,
    translates if necessary, and then generates a personalized response using the full context.
    """
    if st.session_state.get("logged_in"):
        rl_agent = st.session_state.rl_agent
        local_action_modifiers = action_modifiers

        current_user_email = st.session_state["user_info"]["email"]
        font_size, high_contrast = create_sidebar()
        apply_accessibility_styles(font_size, high_contrast)

        st.markdown(f"""
        <div style='margin-bottom: 1.5rem;'>
            <p style='font-size: {font_size}px; margin-top: 1rem;'>
                This Voice assistant is here to help you! It's designed to be:
                <ul style='font-size: {font_size}px;'>
                    <li>Easy to read and understand</li>
                    <li>Patient and supportive</li>
                    <li>Flexible to your needs</li>
                    <li>Emotion-aware</li>
                </ul>
                Speak your request after selecting a language below.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .stButton > button {
            border: none !important;
            box-shadow: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

        if st.session_state.chat_id is not None:
            initialize_chat()
            display_messages(font_size)
            st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)

            # Store the user's selected language
            if "preferred_language" not in st.session_state:
                st.session_state["preferred_language"] = "English"
            st.session_state["preferred_language"] = st.selectbox(
                "Select Language",
                list(languages.keys()),
                index=0,
                key="voice_lang"
            )

            speak_btn = st.button("Speak", help="Press to record", key="voice_btn")

            if speak_btn:
                with st.spinner("Listening..."):
                    recognizer = sr.Recognizer()
                    microphone = sr.Microphone()
                    with microphone as source:
                        recognizer.dynamic_energy_threshold = True
                        recognizer.pause_threshold = 1.4
                        audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)

                try:
                    user_language_code = languages[st.session_state["preferred_language"]]
                    voice_command = recognizer.recognize_google(audio, language=user_language_code)
                    audio_emotion = predict_emotion_text(voice_command)['emotion']
                    save_emotion_to_json(audio_emotion, current_user_email)

                    # Display user's voice input
                    with st.chat_message("user"):
                        st.markdown(f"<p style='font-size: {font_size}px; color: #666;'><em>Emotion: {audio_emotion}</em></p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: {font_size}px;'>{voice_command}</p>", unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "user",
                        "content": voice_command,
                        "emotion": audio_emotion
                    })

                    # Translate to English for semantic lookup
                    english_cmd = translate(voice_command, user_language_code, "en")

                    stored_response = lookup_global_kb(english_cmd)
                    if stored_response:
                        final_response_en = stored_response
                    else:
                        chosen_action = rl_agent.choose_action(audio_emotion)
                        st.session_state.last_action = chosen_action
                        modifier = local_action_modifiers[chosen_action]
                        final_response_en = generate_personalized_response(
                            english_cmd, audio_emotion, current_user_email, modifier
                        )
                        update_global_kb(english_cmd, final_response_en, audio_emotion)

                    # Translate AI response back
                    final_response_local = translate(final_response_en, "en", user_language_code)

                    # Append AI response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_response_local,
                        "avatar": AI_AVATAR_ICON,
                        "emotion": audio_emotion
                    })
                    with st.chat_message("assistant"):
                        st.markdown(f"<p style='font-size: {font_size}px;'>{final_response_local}</p>", unsafe_allow_html=True)

                    # TTS playback
                    play(final_response_local, lang=user_language_code)

                    # Save
                    st.session_state.last_ai_index = len(st.session_state.messages) - 1
                    user_dir = f"data/{current_user_email}"
                    os.makedirs(user_dir, exist_ok=True)
                    joblib.dump(st.session_state.messages, f"{user_dir}/{st.session_state.chat_id}-st_messages")
                    joblib.dump(st.session_state.gemini_history, f"{user_dir}/{st.session_state.chat_id}-gemini_messages")

                    st.rerun()

                except sr.WaitTimeoutError:
                    st.error("No speech detected.")
                except sr.UnknownValueError:
                    st.error("Could not understand audio.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

            # Feedback logic
            if "last_ai_index" in st.session_state and st.session_state.last_ai_index is not None:
                if st.session_state.last_ai_index < len(st.session_state.messages):
                    last_msg = st.session_state.messages[st.session_state.last_ai_index]
                    if last_msg["role"] == "assistant":
                        col1, col2 = st.columns(2)
                        if col1.button("👍", key="voice_like_feedback"):
                            # Positive feedback can still use AI's emotion or user emotion
                            emotion = last_msg.get("emotion") or "neutral"
                            action = st.session_state.get("last_action", 0)
                            rl_agent.update(emotion, action, 1)
                            rl_agent.save(st.session_state["user_info"]["email"])
                            st.success("Thank you for your feedback!")
                            st.session_state.last_ai_index = None
                            st.rerun()

                        if col2.button("👎", key=f"voice_dislike_feedback_{st.session_state.last_ai_index}"):
                            st.error("Feedback noted. Regenerating response...")

                            # Remove disliked AI response
                            st.session_state.messages.pop(st.session_state.last_ai_index)

                            # Find the user's last message (to get user emotion)
                            user_msg = None
                            for i in reversed(range(len(st.session_state.messages))):
                                if st.session_state.messages[i]["role"] == "user":
                                    user_msg = st.session_state.messages[i]
                                    break
                            if not user_msg:
                                st.warning("No user message found to regenerate from.")
                                st.session_state.last_ai_index = None
                                st.rerun()

                            user_emotion = user_msg.get("emotion", "neutral")
                            action = st.session_state.get("last_action", 0)
                            rl_agent.update(user_emotion, action, -1)  # negative feedback
                            rl_agent.save(st.session_state["user_info"]["email"])

                            # Re-generate with user emotion
                            user_language_code = languages[st.session_state["preferred_language"]]
                            english_cmd = translate(user_msg["content"], user_language_code, "en")
                            new_action = rl_agent.choose_action(user_emotion)
                            st.session_state.last_action = new_action
                            new_modifier = local_action_modifiers[new_action]
                            new_response_en = generate_personalized_response(
                                english_cmd, user_emotion,
                                st.session_state["user_info"]["email"],
                                new_modifier
                            )
                            new_response_local = translate(new_response_en, "en", user_language_code)

                            # Append new AI message
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": new_response_local,
                                "avatar": AI_AVATAR_ICON,
                                "emotion": user_emotion
                            })
                            st.session_state.last_ai_index = len(st.session_state.messages) - 1

                            # Save
                            user_dir = f"data/{st.session_state['user_info']['email']}"
                            os.makedirs(user_dir, exist_ok=True)
                            joblib.dump(st.session_state.messages, f"{user_dir}/{st.session_state.chat_id}-st_messages")
                            joblib.dump(st.session_state.gemini_history, f"{user_dir}/{st.session_state.chat_id}-gemini_messages")
                            st.rerun()

        else:
            st.info("Please create or select a conversation from the sidebar.")
    else:
        st.warning("Please login/signup to view the dashboard.")

def home():
    home_css = """
    <style>
        html, body, .stApp {
            background: linear-gradient(to left, #808080, #fbc8c4) !important;
        }
        .container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: flex-start;
            height: 70vh;
            width: 100vw;
        }
        .container h1 {
            font-size: 5rem !important;
            font-weight: bold !important;
            color: #4a4a4a !important;
            text-align: left !important;
            margin-bottom: 20px !important;
            line-height: 1.2 !important;
            font-family: 'Poppins', sans-serif !important;
        }
        .container p {
            font-size: 1.2rem;
            font-weight: bold !important;
            color: #4a4a4a;
            max-width: 600px;
            text-align: left;
            margin-top: 10px;
        }
    </style>
    """
    st.markdown(home_css, unsafe_allow_html=True)
    st.markdown("""
        <div class="container">
            <h1>Automated <br> Emotional <br> Support <br> Solutions</h1>
            <p>Enhancing communication and emotional well-being through intelligent interactions and knowledge applications.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button('Begin'):
        st.session_state.page = "Signup/Login"
        st.rerun()

def reset_styles():
    st.markdown("""
    <style>
        html, body, .stApp {
            background: black !important;
            font-family: sans-serif !important;
        }
    </style>
    """, unsafe_allow_html=True)

def main(json_file_path="data.json"):
    if "initialized" not in st.session_state:
        st.session_state.page = "Home"
        st.session_state.chat_id = None
        st.session_state.messages = []
        st.session_state.gemini_history = []
        st.session_state.initialized = True
    required_states = {
        'page': "Home",
        'logged_in': False,
        'chat_id': None,
        'messages': [],
        'gemini_history': [],
        'initialized': True
    }
    for key, value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    st.sidebar.title("Your Personal Mental Health Expert")
    if st.session_state.logged_in:
        options = ("AI assistant", "Voice Assistant", "Statistics")
    else:
        options = ("Home", "Signup/Login") if st.session_state.page == "Home" else ("Signup/Login",)
    if st.session_state.page not in options:
        st.session_state.page = options[0]
    selected_page = st.sidebar.radio("Go to", options, index=options.index(st.session_state.page), key="pages")
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Signup/Login":
        st.title("Signup/Login ")
        login_or_signup = st.radio("Select an option", ("Login", "Signup"), key="login_signup")
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)
    elif st.session_state.page == "AI assistant":
        if st.session_state.get("logged_in"):
            ai_assistant()
        else:
            st.warning("Please login/signup to view the dashboard.")
    elif st.session_state.page == "Voice Assistant":
        if st.session_state.get("logged_in"):
            voice_assistant()
        else:
            st.warning("Please login/signup to view the dashboard.")
    elif st.session_state.page == "Statistics":
        if st.session_state.get("logged_in") and st.session_state.get("user_info"):
            st.title("Emotion & Mood Statistics")
            current_user_email = st.session_state["user_info"]["email"]
            emotions_data = load_user_emotions(current_user_email, json_file_path)
            if not emotions_data:
                st.warning("No emotion data available.")
            else:
                df = pd.DataFrame(emotions_data)
                df["emotion"] = df["emotion"].str.lower()
                df["emotion"] = df["emotion"].replace({
                    "neutral version": "neutral",
                    "neutral ": "neutral",
                })
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                color_mapping = {
                    'anger': '#FF0000',
                    'anxiety/fear': '#FFA500',
                    'happiness': '#1f77b4',
                    'disgust': '#2ca02c',
                    'boredom': '#9467bd',
                    'sadness': '#8c564b',
                    'neutral': '#7f7f7f'
                }
                st.subheader("Mood Distribution")
                mood_dist = df["emotion"].value_counts().sort_values(ascending=False)
                total = mood_dist.sum()
                legend_labels = [f"{emo}: {(mood_dist[emo] / total * 100):.1f}%" for emo in mood_dist.index]
                colors = [color_mapping.get(emo, "#999999") for emo in mood_dist.index]
                fig, ax = plt.subplots(figsize=(8, 8))
                wedges, texts, autotexts = ax.pie(
                    mood_dist,
                    autopct="",
                    startangle=140,
                    colors=colors,
                    wedgeprops={"width": 0.3, "edgecolor": "white"},
                    pctdistance=0.75
                )
                centre_circle = plt.Circle((0, 0), 0.70, fc="white")
                fig.gca().add_artist(centre_circle)
                ax.axis("equal")
                ax.legend(wedges, legend_labels, title="Emotions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Emotional Trends Over Time")
                df["date"] = df["timestamp"].dt.date
                daily_counts = df.groupby(["date", "emotion"]).size().reset_index(name="count")
                pivoted = daily_counts.pivot(index="date", columns="emotion", values="count").fillna(0)
                used_emotions = [emo for emo in color_mapping.keys() if emo in pivoted.columns]
                pivoted = pivoted[used_emotions]
                line_colors = [color_mapping[emo] for emo in used_emotions]
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                pivoted.plot(ax=ax2, marker="o", color=line_colors)
                ax2.set_title("Emotional Trends Over Time")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Count")
                plt.xticks(rotation=45)
                plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                st.pyplot(fig2)
        else:
            st.warning("Please login to view your statistics.")
    else:
        st.warning("Please login to view the dashboard.")

if __name__ == "__main__":
    main()
