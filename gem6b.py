import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
import emoji
import nltk
from nltk.corpus import stopwords
from fpdf import FPDF
import base64
from statsmodels.tsa.arima.model import ARIMA
import random
from textblob import TextBlob
from datetime import datetime, timedelta
import numpy as np

nltk.download('stopwords')
import bcrypt
import json
import os

# Import Google Generative AI
import google.generativeai as genai

# Configure Gemini API (replace with your actual API key)
# It's highly recommended to use Streamlit secrets for API keys
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.warning("Gemini API key not found in Streamlit secrets. Some AI features may not work. Please add it to .streamlit/secrets.toml or set as an environment variable.")
    # For a quick local test (NOT for production or public code):
    # REMINDER: Replace "YOUR_GEMINI_API_KEY" with your actual Gemini API key for functionality.
    genai.configure(api_key="YOUR_GEMINI_API_KEY")


# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

CREDENTIALS_FILE = "users.json"

def load_users():
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "r") as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump(users, file)

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = hashed_pw
    save_users(users)
    return True

def authenticate_user(username, password):
    users = load_users()
    if username in users and bcrypt.checkpw(password.encode(), users[username].encode()):
        return True
    return False


# ===================== Preprocessing Function =====================
def preprocess_chat(data):
    data = data.replace('\u202f', ' ').replace('\u200e', '')
    pattern = r'(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}) (am|pm) - (.*?): (.*)'
    matches = re.findall(pattern, data, flags=re.IGNORECASE)

    records = []
    media_messages_count = 0
    for date, time, ampm, user, message in matches:
        timestamp = f"{date} {time} {ampm}"
        try:
            dt = pd.to_datetime(timestamp, format='%d/%m/%Y %I:%M %p')
            if message.strip().lower() == "<media omitted>":
                media_messages_count += 1
            elif message.strip().lower() != "null":
                records.append((dt, user.strip(), message.strip()))
        except:
            continue

    df = pd.DataFrame(records, columns=['datetime', 'user', 'message'])
    df.loc[:, 'media_message'] = False # Default to False for all records in the text dataframe
    
    # Store media count separately, or add a conceptual row/metadata
    st.session_state['media_messages_count'] = media_messages_count
    
    return df

# ===================== Analysis Functions =====================
def filter_data(df, selected_users, start_date, end_date):
    if selected_users:
        df = df[df['user'].isin(selected_users)]
    if start_date and end_date:
        df = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]
    return df

def weekly_activity(df):
    df['weekday'] = df['datetime'].dt.day_name()
    return df['weekday'].value_counts().sort_index()

def monthly_activity(df):
    df['month'] = df['datetime'].dt.strftime('%B %Y')
    return df['month'].value_counts().sort_index()

def hourly_distribution(df):
    df['hour'] = df['datetime'].dt.hour
    return df['hour'].value_counts().sort_index()

def heatmap_data(df):
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day_name()
    return df.pivot_table(index='day', columns='hour', values='message', aggfunc='count').fillna(0)

def common_words(df):
    stop_words = set(stopwords.words('english'))
    words = ' '.join(df['message']).lower()
    words = re.findall(r'\b\w+\b', words)
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return Counter(filtered_words).most_common(20)

def user_frequency(df):
    return df['user'].value_counts()

def generate_wordcloud(df):
    text = ' '.join(df['message'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wc

def emoji_counter(df):
    emojis = []
    for msg in df['message']:
        emojis += [c for c in msg if c in emoji.EMOJI_DATA]
    return Counter(emojis).most_common(10)

def user_message_stats(df):
    df['length'] = df['message'].apply(len)
    avg_lengths = df.groupby('user')['length'].mean().sort_values(ascending=False)
    daily_counts = df.groupby(['user', df['datetime'].dt.date]).size().groupby('user').mean()
    return avg_lengths, daily_counts

def sentiment_analysis(df):
    # Ensure 'polarity' column is created within this function or before it's called
    df['polarity'] = df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    user_sentiment = df.groupby('user')['polarity'].mean().sort_values()
    return user_sentiment

def predict_busiest_month(df, steps=3):
    df['month'] = df['datetime'].dt.to_period('M')
    monthly_counts = df.groupby('month').size()
    monthly_counts.index = monthly_counts.index.to_timestamp()
    if len(monthly_counts) < 3:
        return None, None, "Not enough data for a trend."
    model = ARIMA(monthly_counts, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    predicted_month = forecast.idxmax()
    explanation = (
        f"The prediction is based on recent trends. "
        f"The forecast shows the highest expected activity in {predicted_month.strftime('%B %Y')} "
        f"with an estimated {int(forecast.max())} messages."
    )
    return predicted_month.strftime('%B %Y'), forecast, explanation

def generate_future_messages(df, num_sentences=5):
    text = ' '.join(df['message'].dropna())
    words = text.split()
    if len(words) < 2:
        return ["Not enough data to predict future messages."]
    markov_chain = {}
    for i in range(len(words) - 1):
        key, next_word = words[i], words[i + 1]
        markov_chain.setdefault(key, []).append(next_word)
    messages = []
    for _ in range(num_sentences):
        word = random.choice(words)
        sentence = [word]
        for _ in range(random.randint(4, 12)):
            word = random.choice(markov_chain.get(word, words))
            sentence.append(word)
        messages.append(' '.join(sentence).capitalize() + '.')
    return messages

def predict_future_words(df, num_words=10):
    text = ' '.join(df['message'].dropna()).lower()
    words = re.findall(r'\b\w+\b', text)
    if len(words) < 2:
        return ["Not enough data to predict future words."]

    markov_chain = {}
    for i in range(len(words) - 1):
        key = words[i]
        next_word = words[i + 1]
        markov_chain.setdefault(key, []).append(next_word)

    current_word = random.choice(words)
    future_words = [current_word]
    for _ in range(num_words - 1):
        next_words = markov_chain.get(current_word, words)
        current_word = random.choice(next_words)
        future_words.append(current_word)

    return future_words

def predict_active_days(df):
    weekday_counts = df['datetime'].dt.day_name().value_counts()
    top_days = weekday_counts.head(2).index.tolist()
    return top_days, f"Based on past data, most messages are usually sent on {', '.join(top_days)}."

def predict_active_hours(df):
    hour_counts = df['datetime'].dt.hour.value_counts().sort_index()
    peak_hours = hour_counts[hour_counts == hour_counts.max()].index.tolist()
    return peak_hours, f"Most activity has been observed around {', '.join(str(h) + ':00' for h in peak_hours)}."

def detect_message_trend(df):
    df['month'] = df['datetime'].dt.to_period('M')
    monthly = df.groupby('month').size()
    if len(monthly) < 3:
        return "Not enough data to detect trend."
    trend = "increasing" if monthly.iloc[-1] > monthly.iloc[0] else "decreasing"
    return f"Message volume is generally {trend} over time."

# New: Calculate Response Times
def calculate_response_times(df):
    df_sorted = df.sort_values(by='datetime').reset_index(drop=True)
    response_times = []
    
    # Iterate through messages to find responses from different users
    for i in range(1, len(df_sorted)):
        prev_msg = df_sorted.iloc[i-1]
        current_msg = df_sorted.iloc[i]

        if current_msg['user'] != prev_msg['user']:
            time_diff = (current_msg['datetime'] - prev_msg['datetime']).total_seconds()
            response_times.append(time_diff)
            
    if not response_times:
        return None, "Not enough conversations to calculate response times."
    
    avg_response_sec = sum(response_times) / len(response_times)
    
    hours, remainder = divmod(avg_response_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s", response_times

# New: Message Type Distribution
def message_type_distribution(total_text_messages, total_media_messages):
    labels = ['Text Messages', 'Media Messages']
    sizes = [total_text_messages, total_media_messages]
    
    if total_text_messages == 0 and total_media_messages == 0:
        return None, "No messages to display distribution."
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig, None

# New: Top Active Hours List
def top_active_hours_list(df, top_k=5):
    hourly_counts = df['datetime'].dt.hour.value_counts().sort_values(ascending=False)
    if hourly_counts.empty:
        return "No activity data."
    
    top_hours = hourly_counts.head(top_k)
    return top_hours

# ===================== Gemini-powered features =====================

def summarize_chat(chat_text):
    if len(chat_text) > 10000: # Limit input length for API call
        chat_text = chat_text[:10000] + "..." # Truncate and indicate truncation
        st.warning("Chat text too long for full summarization. Summarizing truncated text.")
    try:
        prompt = f"Summarize the following WhatsApp chat conversation. Focus on key topics and overall sentiment:\n\n{chat_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error summarizing chat: {e}"

def generate_smart_reply(last_message):
    try:
        prompt = f"Given the last message in a WhatsApp chat: '{last_message}', suggest a concise and relevant smart reply."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating smart reply: {e}"

# Modified: generate_message_content with length control, emoji option, and robust output cleaning
def generate_message_content(topic, tone="neutral", length="short", add_emojis=False):
    length_prompt = ""
    if length == "short":
        length_prompt = "Keep it concise and brief."
    elif length == "medium":
        length_prompt = "Aim for a moderate length, around 2-3 sentences."
    elif length == "long":
        length_prompt = "Make it detailed, around 4-5 sentences."
    elif length == "very long":
        length_prompt = "Elaborate extensively, providing comprehensive information."

    emoji_prompt = ""
    if add_emojis:
        emoji_prompt = " Include relevant and expressive emojis to make it more vibrant."

    try:
        # Strengthen the prompt to strictly output only the message
        prompt = f"Generate a WhatsApp message about '{topic}' in a '{tone}' tone. {length_prompt}{emoji_prompt} Output ONLY the message text. Do NOT include any explanations, introductions, or conversational filler."
        response = model.generate_content(prompt)
        
        # Post-process to remove any unwanted explanations or meta-text that the model might still add
        generated_text = response.text
        
        # Regex to remove common patterns of unwanted explanations
        # This will remove anything from "Explanation of choices" or "Explanation:" onwards
        # It also handles variations of "Best," or similar sign-offs followed by explanations
        generated_text = re.split(r'(\n+Explanation of choices|\n+Explanation:|\n*Best, .*?\n*Explanation of choices|\n*Best, .*?\n*Explanation:)', generated_text, 1)[0]
        generated_text = generated_text.strip() # Remove leading/trailing whitespace

        # Also remove common conversational filler at the start/end if present
        if generated_text.lower().startswith("here's a "):
            generated_text = re.sub(r"^[Hh]ere's a (generated |suggested |concise )?whatsapp message( for you)?:?\s*", "", generated_text, flags=re.IGNORECASE)
        if generated_text.lower().endswith("hope this helps."):
            generated_text = re.sub(r"hope this helps\.$", "", generated_text, flags=re.IGNORECASE)

        return generated_text.strip()
    except Exception as e:
        return f"Error generating message content: {e}"

def identify_key_topics(chat_text):
    if len(chat_text) > 10000: # Limit input length for API call
        chat_text = chat_text[:10000] + "..."
        st.warning("Chat text too long for full topic identification. Identifying topics from truncated text.")
    try:
        prompt = f"Identify the main topics or themes discussed in the following WhatsApp chat conversation. List them concisely.\n\n{chat_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error identifying topics: {e}"

# New: Function to generate a meeting summary
def generate_meeting_summary(chat_text):
    if len(chat_text) > 10000:
        chat_text = chat_text[:10000] + "..."
        st.warning("Chat text too long for full meeting summarization. Summarizing truncated text.")
    try:
        prompt = f"Given the following chat, identify if it contains a meeting discussion. If so, summarize the key decisions, action items, and participants. If not, state that it does not appear to be a meeting chat:\n\n{chat_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating meeting summary: {e}"

# New: Function for sentiment trend over time
def sentiment_trend(df):
    df['date'] = df['datetime'].dt.date
    daily_sentiment = df.groupby('date')['polarity'].mean().reset_index()
    return daily_sentiment

# New: Function to identify most active hours by day of week
def active_hours_by_day(df):
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    active_pivot = df.pivot_table(index='day_of_week', columns='hour', values='message', aggfunc='count').fillna(0)
    # Order days of week
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    active_pivot = active_pivot.reindex(ordered_days, fill_value=0)
    return active_pivot

# New: Function to identify most common phrases
def common_phrases(df, n=2, top_k=10):
    text = ' '.join(df['message'].dropna()).lower()
    words = re.findall(r'\b\w+\b', text)
    if len(words) < n:
        return []

    phrases = []
    for i in range(len(words) - n + 1):
        phrases.append(tuple(words[i:i+n]))

    return Counter(phrases).most_common(top_k)

# New: Function to identify recurring patterns/habits (basic example)
def identify_chat_habits(df):
    habits = []
    # Most active user
    most_active_user = df['user'].value_counts().idxmax()
    habits.append(f"Most active participant: **{most_active_user}**.")

    # Peak hour for messages
    peak_hour = df['datetime'].dt.hour.mode()[0]
    habits.append(f"Peak messaging hour: **{peak_hour}:00** to **{peak_hour+1}:00**.")

    # Most common day
    most_common_day = df['datetime'].dt.day_name().mode()[0]
    habits.append(f"Most active day of the week: **{most_common_day}**.")

    # Basic check for consistent morning/evening activity
    morning_msgs = df[(df['datetime'].dt.hour >= 6) & (df['datetime'].dt.hour < 12)]
    evening_msgs = df[(df['datetime'].dt.hour >= 18) & (df['datetime'].dt.hour < 24)]
    if len(morning_msgs) > len(df) * 0.3:
        habits.append("Consistent morning activity detected.")
    if len(evening_msgs) > len(df) * 0.3:
        habits.append("Consistent evening activity detected.")

    return habits

# New: Function to detect conversation starters
def detect_conversation_starters(df, top_k=5):
    starters = []
    # Using the first messages by each user on a given day/hour as potential starters
    df_sorted = df.sort_values(by='datetime')
    df_sorted['prev_user'] = df_sorted['user'].shift(1)
    df_sorted['time_diff_minutes'] = (df_sorted['datetime'] - df_sorted['datetime'].shift(1)).dt.total_seconds() / 60

    potential_starters = df_sorted[
        (df_sorted['time_diff_minutes'].isna()) | # First message
        (df_sorted['time_diff_minutes'] > 30) |    # Long pause (configurable threshold)
        (df_sorted['datetime'].dt.date != df_sorted['datetime'].shift(1).dt.date) # New day
    ]

    # Filter out media omitted or short/empty messages
    potential_starters = potential_starters[
        (potential_starters['message'].str.strip().str.len() > 5) &
        (~potential_starters['message'].str.contains(r'<media omitted>', case=False))
    ]

    if len(potential_starters) == 0:
        return ["Not enough distinct conversation starters detected."]

    return potential_starters['message'].value_counts().head(top_k).index.tolist()

# New: Analyze Overall Chat Tone/Mood
def analyze_overall_chat_tone(chat_text):
    if len(chat_text) > 10000:
        chat_text = chat_text[:10000] + "..."
        st.warning("Chat text too long for full tone analysis. Analyzing truncated text.")
    try:
        prompt = f"Analyze the overall tone or mood of the following WhatsApp chat conversation. Is it generally positive, negative, neutral, humorous, formal, informal, serious, casual, etc.? Provide a concise description.\n\n{chat_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing chat tone: {e}"

# NEW: Love Score Analysis (AI)
def calculate_love_score(chat_df):
    if chat_df.empty:
        return "Not enough chat data to calculate a love score. Please upload a valid chat.", None, None, None

    # Get overall chat text
    full_chat_text = ' '.join(chat_df['message'].dropna().tolist())
    
    # Calculate some basic metrics that can be included in the prompt
    num_users = chat_df['user'].nunique()
    total_messages = len(chat_df)
    
    if num_users != 2:
        return (f"This chat has {num_users} participants. For an accurate 'love score' based on couple dynamics, please upload a chat with only two participants.", None, None, None)

    # Calculate sentiment
    chat_df['polarity'] = chat_df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    avg_polarity = chat_df['polarity'].mean()

    # Communication balance
    user_message_counts = chat_df['user'].value_counts()
    user1, user2 = user_message_counts.index[0], user_message_counts.index[1]
    count1, count2 = user_message_counts.iloc[0], user_message_counts.iloc[1]
    balance = min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 0

    # Common words (top 10 to include in prompt)
    common_w = common_words(chat_df)[:10]
    
    # Emojis (top 5 to include in prompt)
    common_e = emoji_counter(chat_df)[:5]

    # Length consideration for prompt
    prompt_chat_text = full_chat_text
    if len(full_chat_text) > 8000: # Slightly larger limit for love score, but still cautious
        prompt_chat_text = full_chat_text[:8000] + "..." # Truncate for prompt
        st.warning("Chat text too long for full love score analysis. Analyzing truncated text.")

    # Constructing a detailed prompt for Gemini
    prompt = f"""
    You are an AI relationship analyst. Analyze the following WhatsApp chat conversation between two individuals. 
    Your goal is to assess the relationship's "love score" on a scale of 1 to 100, where 100 is highly loving and connected, and 1 is distant or problematic.

    Consider the following aspects of the chat:
    - **Overall Sentiment:** The general emotional tone of messages. (Average polarity: {avg_polarity:.2f})
    - **Communication Balance:** How evenly do the two participants contribute to the conversation? (Message ratio of {user1}: {count1} to {user2}: {count2}. Balance score: {balance:.2f})
    - **Frequency of Communication:** How consistently do they interact? (Total messages: {total_messages})
    - **Use of Affectionate Language/Emojis:** Presence of terms of endearment, positive words, and love-related emojis. (Common words: {common_w}, Common emojis: {common_e})
    - **Shared Interests/Topics:** Do they discuss common topics or support each other's interests?
    - **Empathy and Support:** Do messages show understanding, care, and mutual encouragement?
    - **Conflict Resolution (if apparent):** How are disagreements handled? (If any are present in the provided text.)
    - **Engagement:** Are responses timely and thoughtful, or short and dismissive?

    The chat conversation is as follows:
    ---
    {prompt_chat_text}
    ---

    Based on your analysis, provide:
    1.  **A Love Score (out of 100):** A numerical value.
    2.  **A Detailed Explanation:** Justify your score by explicitly referencing the positive and negative aspects observed in the chat, based on the criteria above. Be specific.
    3.  **Recommendations (Optional):** Suggest areas for improvement or ways to foster an even stronger connection, if applicable.

    Format your output clearly:
    Love Score: [SCORE]/100
    Explanation: [Your detailed explanation]
    Recommendations: [Your recommendations or "N/A"]
    """

    try:
        response = model.generate_content(prompt)
        # Attempt to parse the response
        response_text = response.text
        
        score_match = re.search(r"Love Score:\s*(\d+)/100", response_text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else None
        
        explanation_match = re.search(r"Explanation:\s*(.*?)(?=\nRecommendations:|\nLove Score:|$)", response_text, re.IGNORECASE | re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else "Could not extract explanation."
        
        recommendations_match = re.search(r"Recommendations:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
        recommendations = recommendations_match.group(1).strip() if recommendations_match else "N/A"

        return "Success", score, explanation, recommendations

    except Exception as e:
        return f"Error calculating love score: {e}", None, None, None


# NEW FEATURES FOR LOVE SCORE SECTION

def calculate_relationship_timeline_insights(df):
    if df.empty:
        return None, None, None, None

    first_msg_date = df['datetime'].min().date()
    last_msg_date = df['datetime'].max().date()
    
    # Corrected variable name: days_active instead of total_days_chatted
    days_active = (last_msg_date - first_msg_date).days + 1

    # Longest gap between replies (between different users)
    df_sorted = df.sort_values(by='datetime').reset_index(drop=True)
    longest_gap = timedelta(0)
    
    # Iterate through messages to find responses from different users
    if len(df_sorted) > 1:
        for i in range(1, len(df_sorted)):
            prev_msg = df_sorted.iloc[i-1]
            current_msg = df_sorted.iloc[i]

            if current_msg['user'] != prev_msg['user']:
                time_diff = current_msg['datetime'] - prev_msg['datetime']
                if time_diff > longest_gap:
                    longest_gap = time_diff
    
    # Streaks of consistent daily chatting
    active_dates = df['datetime'].dt.date.unique()
    active_dates.sort()
    
    longest_streak = 0
    current_streak = 0
    if len(active_dates) > 0:
        current_streak = 1
        longest_streak = 1
        for i in range(1, len(active_dates)):
            if (active_dates[i] - active_dates[i-1]).days == 1:
                current_streak += 1
            else:
                current_streak = 1
            longest_streak = max(longest_streak, current_streak)

    return first_msg_date, days_active, longest_gap, longest_streak

def get_conversation_balance(df):
    if df.empty:
        return None, None, None, None, None

    user_message_counts = df['user'].value_counts()
    
    # Chat dominance ratio
    if len(user_message_counts) == 2:
        user1_name, user2_name = user_message_counts.index[0], user_message_counts.index[1]
        user1_count, user2_count = user_message_counts.iloc[0], user_message_counts.iloc[1]
        
        ratio = f"{user1_name}: {user1_count}, {user2_name}: {user2_count}"
        dominance_text = f"The message distribution is {user1_name} ({user1_count} messages) and {user2_name} ({user2_count} messages)."
        
        # Who sends longer messages
        df['length'] = df['message'].apply(len)
        avg_lengths = df.groupby('user')['length'].mean()
        longer_msgs_user = avg_lengths.idxmax()
        longer_msgs_avg = avg_lengths.max()
        
        # Who initiates more messages (first message after a 30 min idle period or new day)
        df_sorted = df.sort_values(by='datetime').reset_index(drop=True)
        initiator_counts = Counter()
        
        if not df_sorted.empty:
            initiator_counts[df_sorted.iloc[0]['user']] += 1 # First message overall
            for i in range(1, len(df_sorted)):
                time_diff = (df_sorted.iloc[i]['datetime'] - df_sorted.iloc[i-1]['datetime']).total_seconds() / 60 # in minutes
                if time_diff > 30 or df_sorted.iloc[i]['datetime'].date() != df_sorted.iloc[i-1]['datetime'].date():
                    initiator_counts[df_sorted.iloc[i]['user']] += 1
        
        if initiator_counts:
            most_initiator = initiator_counts.most_common(1)[0][0]
            initiator_text = f"**{most_initiator}** seems to initiate conversations more often."
        else:
            initiator_text = "Could not determine conversation initiators."

        return dominance_text, longer_msgs_user, longer_msgs_avg, initiator_text, user_message_counts
    else:
        return "Not a two-person chat.", None, None, None, user_message_counts # Return user_message_counts even if not 2 people

def top_romantic_messages(df, num_messages=5):
    if df.empty:
        return pd.DataFrame()
    
    df['polarity'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    most_romantic = df[df['polarity'] > 0.3].sort_values(by='polarity', ascending=False)
    
    # Filter out short messages or media omitted placeholders
    most_romantic = most_romantic[most_romantic['message'].str.strip().str.len() > 10]
    most_romantic = most_romantic[~most_romantic['message'].str.contains(r'<media omitted>', case=False)]

    return most_romantic[['datetime', 'user', 'message', 'polarity']].head(num_messages)

def calculate_response_times_per_user(df):
    if df.empty or df['user'].nunique() < 2:
        return "Not enough data for per-user response times.", {}, None

    df_sorted = df.sort_values(by='datetime').reset_index(drop=True)
    user_response_times = {user: [] for user in df['user'].unique()}
    
    # Map user to their last message time
    last_message_time = {user: None for user in df['user'].unique()}

    for i in range(len(df_sorted)):
        current_msg = df_sorted.iloc[i]
        current_user = current_msg['user']
        current_time = current_msg['datetime']

        # Update last message time for the current user
        last_message_time[current_user] = current_time

        # To calculate current_user's response time to previous message (from other user)
        if i > 0 and df_sorted.iloc[i-1]['user'] != current_user: # If the previous message was from the *other* user
            response_diff = current_time - df_sorted.iloc[i-1]['datetime']
            user_response_times[current_user].append(response_diff.total_seconds())

    avg_response_sec_per_user = {user: (sum(times) / len(times)) if times else 0 for user, times in user_response_times.items()}
    
    # Convert to readable format
    avg_response_readable = {}
    for user, seconds in avg_response_sec_per_user.items():
        if seconds > 0:
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds_rem = divmod(remainder, 60)
            avg_response_readable[user] = f"{int(hours)}h {int(minutes)}m {int(seconds_rem)}s"
        else:
            avg_response_readable[user] = "N/A (No responses detected)"

    fastest_replier = None
    min_avg_time = float('inf')
    for user, avg_sec in avg_response_sec_per_user.items():
        if avg_sec > 0 and avg_sec < min_avg_time:
            min_avg_time = avg_sec
            fastest_replier = user

    return avg_response_readable, avg_response_sec_per_user, fastest_replier


# New: Personality Tags (AI-based Guess) - Renamed to guess_personality_traits for MBTI-like
def guess_personality_traits(chat_df):
    if chat_df.empty or chat_df['user'].nunique() != 2:
        return "Please upload a two-person chat for personality insights."

    user_stats = {}
    
    for user in chat_df['user'].unique():
        user_df = chat_df[chat_df['user'] == user].copy()
        user_text = ' '.join(user_df['message'].dropna().tolist())
        
        # Calculate user-specific metrics
        num_messages = len(user_df)
        avg_sentiment = user_df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity).mean()
        avg_length = user_df['message'].apply(len).mean()
        
        common_w = ", ".join([word for word, count in common_words(user_df)[:5]])
        common_e = ", ".join([e for e, count in emoji_counter(user_df)[:3]])

        user_stats[user] = {
            "num_messages": num_messages,
            "avg_sentiment": avg_sentiment,
            "avg_length": avg_length,
            "common_words": common_w,
            "common_emojis": common_e,
            "sample_messages": " ".join(user_df['message'].sample(min(5, len(user_df))).tolist())
        }
    
    user1_name, user2_name = list(user_stats.keys())
    user1_stats = user_stats[user1_name]
    user2_stats = user_stats[user2_name]

    prompt = f"""
    You are an AI personality profiler. Analyze the messaging styles of two individuals in a WhatsApp chat.
    Based on the provided statistics and sample messages, infer potential personality traits for each person, similar to a light MBTI-like guess (e.g., "warm and expressive", "reserved and analytical", "spontaneous and energetic"). You can also suggest a relevant emoji or symbol for their style.

    Here are the statistics for {user1_name}:
    - Total Messages: {user1_stats['num_messages']}
    - Avg. Sentiment: {user1_stats['avg_sentiment']:.2f}
    - Avg. Message Length: {user1_stats['avg_length']:.2f} characters
    - Common Words: {user1_stats['common_words']}
    - Common Emojis: {user1_stats['common_emojis']}
    - Sample Messages: "{user1_stats['sample_messages']}"

    Here are the statistics for {user2_name}:
    - Total Messages: {user2_stats['num_messages']}
    - Avg. Sentiment: {user2_stats['avg_sentiment']:.2f}
    - Avg. Message Length: {user2_stats['avg_length']:.2f} characters
    - Common Words: {user2_stats['common_words']}
    - Common Emojis: {user2_stats['common_emojis']}
    - Sample Messages: "{user2_stats['sample_messages']}"

    Based on this, provide a personality description for each user. Format your output as:
    **{user1_name}:** [Personality Description with emoji]
    **{user2_name}:** [Personality Description with emoji]
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error guessing personality traits: {e}"

# New: Conflict & Recovery Patterns
def detect_conflict_and_recovery(df, time_window_minutes=60, sentiment_threshold=-0.1, recovery_threshold=0.2):
    if df.empty or df['user'].nunique() < 2:
        return "Not enough data for conflict detection or not a two-person chat."

    df['polarity'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_sorted = df.sort_values(by='datetime').reset_index(drop=True)
    conflict_patterns = []

    for i in range(len(df_sorted)):
        current_msg = df_sorted.iloc[i]
        
        # Look for negative sentiment
        if current_msg['polarity'] < sentiment_threshold:
            # Define a time window after this negative message
            end_window_time = current_msg['datetime'] + timedelta(minutes=time_window_minutes)
            
            # Check for subsequent messages within the window
            subsequent_msgs = df_sorted[
                (df_sorted['datetime'] > current_msg['datetime']) &
                (df_sorted['datetime'] <= end_window_time) &
                (df_sorted.index > i) # Ensure we look forward from the current message
            ]

            if not subsequent_msgs.empty:
                # Check for significant positive shift within the window
                max_pos_polarity_in_window = subsequent_msgs['polarity'].max()
                if max_pos_polarity_in_window > recovery_threshold and (max_pos_polarity_in_window - current_msg['polarity']) > recovery_threshold:
                    # Found a potential recovery
                    recovery_msg = subsequent_msgs.loc[subsequent_msgs['polarity'].idxmax()]
                    conflict_patterns.append({
                        "start_time": current_msg['datetime'],
                        "end_time": recovery_msg['datetime'],
                        "conflict_message": current_msg['message'],
                        "recovery_message": recovery_msg['message'],
                        "time_to_recover": recovery_msg['datetime'] - current_msg['datetime']
                    })
    
    if conflict_patterns:
        result = "Detected potential conflict and recovery patterns:\n"
        for pattern in conflict_patterns:
            result += f"- Conflict identified around {pattern['start_time'].strftime('%Y-%m-%d %H:%M')} with message: \"{pattern['conflict_message']}\"\n"
            result += f"  Recovery message around {pattern['end_time'].strftime('%Y-%m-%d %H:%M')} with message: \"{pattern['recovery_message']}\"\n"
            result += f"  Time to recovery: {pattern['time_to_recover']}\n\n"
        return result
    else:
        return "No clear conflict and recovery patterns detected based on sentiment swings."

# New: Compatibility Radar Chart
def create_compatibility_radar_chart(df):
    if df.empty or df['user'].nunique() != 2:
        return None, "Please upload a two-person chat to generate a compatibility radar chart."

    users = df['user'].unique()
    if len(users) != 2: # Redundant check but good for safety
        return None, "Radar chart is best for comparing two users. This chat has more/less than two participants."

    user1, user2 = users[0], users[1]

    # Calculate metrics for each user
    df['polarity'] = df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    df['length'] = df['message'].apply(len)

    metrics = {}
    for user in users:
        user_df = df[df['user'] == user]
        metrics[user] = {
            'avg_sentiment': user_df['polarity'].mean(),
            'msg_frequency': len(user_df),
            'avg_msg_length': user_df['length'].mean(),
            'avg_daily_messages': user_df.groupby(user_df['datetime'].dt.date).size().mean() if not user_df.empty else 0
        }

    # Normalize metrics for radar chart (0 to 1 scale)
    # Define max values for normalization - these can be adjusted based on typical chat data
    # Use max values from *both* users or a reasonable global max
    all_msg_frequencies = [metrics[u]['msg_frequency'] for u in users]
    all_avg_msg_lengths = [metrics[u]['avg_msg_length'] for u in users]
    all_avg_daily_messages = [metrics[u]['avg_daily_messages'] for u in users]

    max_msg_frequency = max(all_msg_frequencies) * 1.2 if all_msg_frequencies else 1 # Add buffer
    max_avg_msg_length = max(all_avg_msg_lengths) * 1.2 if all_avg_msg_lengths else 1
    max_avg_daily_messages = max(all_avg_daily_messages) * 1.2 if all_avg_daily_messages else 1

    radar_values_user1 = []
    radar_values_user2 = []

    # Axes for Radar Chart
    labels = [
        'Positivity',
        'Message Frequency',
        'Message Length',
        'Daily Consistency'
    ]

    # Positivity (already -1 to 1, normalize to 0-1)
    radar_values_user1.append((metrics[user1]['avg_sentiment'] + 1) / 2)
    radar_values_user2.append((metrics[user2]['avg_sentiment'] + 1) / 2)

    # Message Frequency (normalized by max frequency in chat)
    radar_values_user1.append(metrics[user1]['msg_frequency'] / max_msg_frequency)
    radar_values_user2.append(metrics[user2]['msg_frequency'] / max_msg_frequency)

    # Message Length (normalized by overall max msg length)
    radar_values_user1.append(metrics[user1]['avg_msg_length'] / max_avg_msg_length)
    radar_values_user2.append(metrics[user2]['avg_msg_length'] / max_avg_msg_length)

    # Daily Consistency (normalized by max avg daily messages)
    radar_values_user1.append(metrics[user1]['avg_daily_messages'] / max_avg_daily_messages)
    radar_values_user2.append(metrics[user2]['avg_daily_messages'] / max_avg_daily_messages)

    # Make the plot circular
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)] # Use np.pi
    angles += angles[:1] # Complete the circle

    radar_values_user1 += radar_values_user1[:1]
    radar_values_user2 += radar_values_user2[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, radar_values_user1, linewidth=2, linestyle='solid', label=user1, color='blue', alpha=0.7)
    ax.fill(angles, radar_values_user1, color='blue', alpha=0.25)

    ax.plot(angles, radar_values_user2, linewidth=2, linestyle='solid', label=user2, color='red', alpha=0.7)
    ax.fill(angles, radar_values_user2, color='red', alpha=0.25)

    ax.set_theta_offset(np.pi / 2) # Use np.pi
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1) # Normalized scale

    ax.set_title('Relationship Compatibility Radar', size=16, color='grey', y=1.1)
    ax.legend(loc='upper right')

    return fig, None

# ===================== Streamlit UI =====================
def show_login_page():
    st.title("Login / Register")
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button("Login")
        with col2:
            register_page_button = st.form_submit_button("Go to Register")

        if login_button:
            if authenticate_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        elif register_page_button:
            st.session_state['page'] = 'register'
            st.experimental_rerun()

    st.subheader("Register")
    with st.form("register_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        register_button = st.form_submit_button("Register")
        if register_button:
            if register_user(new_username, new_password):
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Username already exists.")

def show_main_app():
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.experimental_rerun()

    st.title("🤖 WhatsApp Chat Analyzer 💬")

    uploaded_file = st.file_uploader("Upload your WhatsApp chat (.txt) file", type=["txt"])

    if uploaded_file is not None:
        raw_data = uploaded_file.read().decode("utf-8")
        st.session_state['df'] = preprocess_chat(raw_data)
        st.success("Chat file processed successfully!")

    if 'df' in st.session_state and not st.session_state['df'].empty:
        df = st.session_state['df']
        all_users = ['All Users'] + list(df['user'].unique())
        
        st.sidebar.header("Filter Data")
        selected_users = st.sidebar.multiselect("Select Users", options=all_users, default=['All Users'])
        
        if 'All Users' in selected_users:
            selected_users_for_filter = list(df['user'].unique())
        else:
            selected_users_for_filter = selected_users

        min_date = df['datetime'].min().date()
        max_date = df['datetime'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = date_range[0], date_range[1]
        else:
            start_date, end_date = None, None # Or handle single date selection as needed

        filtered_df = filter_data(df.copy(), selected_users_for_filter, start_date, end_date)
        
        if filtered_df.empty:
            st.warning("No data matches the selected filters. Please adjust your selections.")
            return

        st.subheader("Filtered Chat Data Preview")
        if st.checkbox("Show Raw Filtered Data"):
            st.dataframe(filtered_df)

        total_text_messages = len(filtered_df)
        total_media_messages = st.session_state.get('media_messages_count', 0)
        
        st.markdown(f"Total Messages (Text): **{total_text_messages}**")
        st.markdown(f"Total Messages (Media Omitted): **{total_media_messages}**")

        st.markdown("---")
        st.header("📊 Chat Analysis & Visualization")

        st.subheader("Message Type Distribution")
        fig_msg_type, msg_type_error = message_type_distribution(total_text_messages, total_media_messages)
        if fig_msg_type:
            st.pyplot(fig_msg_type)
        else:
            st.info(msg_type_error)
        
        st.subheader("User-Specific Statistics")
        avg_lengths, daily_counts = user_message_stats(filtered_df)
        st.write("Average Message Length per User (characters):")
        st.dataframe(avg_lengths)
        st.write("Average Messages per Day per User:")
        st.dataframe(daily_counts)

        st.subheader("Sentiment Analysis")
        user_sentiment = sentiment_analysis(filtered_df.copy()) # Pass a copy to avoid modifying original df state
        st.write("Average Sentiment Polarity per User (1.0 = most positive, -1.0 = most negative):")
        st.dataframe(user_sentiment)

        st.subheader("Most Positive and Negative Messages")
        positive_messages = filtered_df[filtered_df['polarity'] > 0.5].sort_values(by='polarity', ascending=False).head(5)
        negative_messages = filtered_df[filtered_df['polarity'] < -0.2].sort_values(by='polarity', ascending=True).head(5)

        if not positive_messages.empty:
            st.write("Top 5 Most Positive Messages:")
            for idx, row in positive_messages.iterrows():
                st.info(f"**{row['user']}** ({row['datetime'].strftime('%Y-%m-%d %H:%M')}): {row['message']} (Polarity: {row['polarity']:.2f})")
        else:
            st.info("No significantly positive messages found in the filtered data.")

        if not negative_messages.empty:
            st.write("Top 5 Most Negative Messages:")
            for idx, row in negative_messages.iterrows():
                st.warning(f"**{row['user']}** ({row['datetime'].strftime('%Y-%m-%d %H:%M')}): {row['message']} (Polarity: {row['polarity']:.2f})")
        else:
            st.info("No significantly negative messages found in the filtered data.")

        st.subheader("Sentiment Trend Over Time")
        daily_sentiment = sentiment_trend(filtered_df.copy())
        if not daily_sentiment.empty:
            fig_sentiment_trend = plt.figure(figsize=(10, 6))
            plt.plot(daily_sentiment['date'], daily_sentiment['polarity'])
            plt.title('Average Sentiment Polarity Over Time')
            plt.xlabel('Date')
            plt.ylabel('Average Polarity')
            plt.grid(True)
            st.pyplot(fig_sentiment_trend)
        else:
            st.info("Not enough data to show sentiment trend.")

        st.subheader("Overall Chat Tone/Mood (AI Analysis)")
        full_filtered_chat_text = ' '.join(filtered_df['message'].dropna().tolist())
        if full_filtered_chat_text:
            with st.spinner("Analyzing overall chat tone..."):
                overall_tone = analyze_overall_chat_tone(full_filtered_chat_text)
                st.write(overall_tone)
        else:
            st.info("No text available to analyze overall chat tone.")

        st.subheader("Multi-User Comparison: Total Messages")
        user_msg_counts = user_frequency(filtered_df)
        fig_user_counts = plt.figure(figsize=(10, 6))
        sns.barplot(x=user_msg_counts.index, y=user_msg_counts.values)
        plt.title('Total Messages by User')
        plt.xlabel('User')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_user_counts)
        
        st.subheader("Activity Breakdown")
        
        st.write("Weekly Activity:")
        weekly_counts = weekly_activity(filtered_df)
        fig_weekly = plt.figure(figsize=(8, 5))
        sns.barplot(x=weekly_counts.index, y=weekly_counts.values,
                    order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        plt.title('Messages by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Messages')
        st.pyplot(fig_weekly)

        st.write("Monthly Activity:")
        monthly_counts = monthly_activity(filtered_df)
        fig_monthly = plt.figure(figsize=(12, 6))
        sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker='o')
        plt.title('Messages by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_monthly)

        st.write("Hourly Activity:")
        hourly_counts = hourly_distribution(filtered_df)
        fig_hourly = plt.figure(figsize=(10, 6))
        sns.lineplot(x=hourly_counts.index, y=hourly_counts.values, marker='o')
        plt.title('Messages by Hour of Day')
        plt.xlabel('Hour of Day (24-hour format)')
        plt.ylabel('Number of Messages')
        plt.xticks(range(0, 24))
        plt.grid(True)
        st.pyplot(fig_hourly)

        st.write("Activity Heatmap (Messages by Hour and Day):")
        heatmap_df = heatmap_data(filtered_df)
        fig_heatmap = plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_df, cmap='viridis', annot=True, fmt=".0f", linewidths=.5)
        plt.title('Message Activity Heatmap')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        st.pyplot(fig_heatmap)

        st.subheader("Top Active Hours by Day of Week")
        active_hours_table = active_hours_by_day(filtered_df)
        st.dataframe(active_hours_table)

        st.subheader("Text Analysis")
        st.write("Most Common Words (excluding stopwords):")
        st.dataframe(pd.DataFrame(common_words(filtered_df), columns=['Word', 'Count']))

        st.write("Word Cloud:")
        wordcloud_img = generate_wordcloud(filtered_df)
        fig_wordcloud = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_img, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig_wordcloud)

        st.write("Top Emojis Used:")
        emojis_df = pd.DataFrame(emoji_counter(filtered_df), columns=['Emoji', 'Count'])
        if not emojis_df.empty:
            st.dataframe(emojis_df)
            fig_emojis = plt.figure(figsize=(8, 5))
            sns.barplot(x='Emoji', y='Count', data=emojis_df)
            plt.title('Top Emojis Used')
            st.pyplot(fig_emojis)
        else:
            st.info("No emojis found in the filtered data.")
        
        st.subheader("Conversation Starters (AI Analysis)")
        starters = detect_conversation_starters(filtered_df)
        if starters:
            st.write("Potential conversation starting messages:")
            for s in starters:
                st.write(f"- \"{s}\"")
        else:
            st.info(starters[0] if isinstance(starters, list) else starters)

        st.subheader("Most Common Phrases (2-word)")
        phrases = common_phrases(filtered_df, n=2)
        if phrases:
            st.dataframe(pd.DataFrame(phrases, columns=['Phrase', 'Count']))
        else:
            st.info("Not enough data to detect common phrases.")
        
        st.subheader("Recurring Chat Habits (AI Analysis)")
        chat_habits = identify_chat_habits(filtered_df)
        for habit in chat_habits:
            st.write(f"- {habit}")

        st.markdown("---")
        st.header("🔮 Predictive Insights")

        st.subheader("Predict Busiest Month (ARIMA Model)")
        busiest_month, forecast_data, explanation = predict_busiest_month(filtered_df.copy())
        if busiest_month:
            st.success(f"The busiest month predicted is: **{busiest_month}**")
            st.info(explanation)
            if forecast_data is not None:
                st.write("Monthly Forecast:")
                st.dataframe(forecast_data.to_frame(name='Predicted Messages'))
        else:
            st.info(explanation)

        st.subheader("Predict Future Messages (AI Generated)")
        num_sentences_to_generate = st.slider("Number of future messages to generate:", 1, 10, 3)
        if st.button("Generate Future Messages"):
            with st.spinner("Generating..."):
                future_messages = generate_future_messages(filtered_df)
                for i, msg in enumerate(future_messages[:num_sentences_to_generate]):
                    st.write(f"**Predicted Message {i+1}:** {msg}")
        
        st.subheader("Predict Future Words (AI Generated)")
        num_words_to_generate = st.slider("Number of future words to predict:", 5, 20, 10)
        if st.button("Predict Future Words"):
            with st.spinner("Predicting..."):
                future_words = predict_future_words(filtered_df)
                st.write(f"**Predicted Word Sequence:** {' '.join(future_words[:num_words_to_generate])}")


        st.subheader("Predict Most Active Days")
        active_days, active_days_msg = predict_active_days(filtered_df)
        st.success(active_days_msg)

        st.subheader("Predict Most Active Hours")
        active_hours, active_hours_msg = predict_active_hours(filtered_df)
        st.success(active_hours_msg)

        st.subheader("Message Volume Trend")
        trend_message = detect_message_trend(filtered_df)
        st.info(trend_message)

        st.markdown("---")
        st.header("❤️ Relationship Insights (Beta) ❤️")
        st.info("These insights are best for chats with only two participants.")

        num_chat_participants = filtered_df['user'].nunique()
        if num_chat_participants == 2:
            st.subheader("Relationship Timeline Insights")
            first_msg_date, days_active, longest_gap, longest_streak = calculate_relationship_timeline_insights(filtered_df)
            if first_msg_date:
                st.write(f"First message date: **{first_msg_date.strftime('%Y-%m-%d')}**")
                st.write(f"Total days with activity: **{days_active} days**")
                st.write(f"Longest gap between replies: **{longest_gap}**")
                st.write(f"Longest consecutive daily chatting streak: **{longest_streak} days**")
            else:
                st.info("Not enough data to determine relationship timeline insights.")

            st.subheader("Conversation Balance")
            balance_text, longer_msgs_user, longer_msgs_avg, initiator_text, user_message_counts = get_conversation_balance(filtered_df)
            st.write(balance_text)
            if longer_msgs_user:
                st.write(f"**{longer_msgs_user}** tends to send longer messages (avg. {longer_msgs_avg:.0f} chars).")
            st.write(initiator_text)
            
            st.subheader("Response Times Per User")
            avg_response_readable, avg_response_sec_per_user, fastest_replier = calculate_response_times_per_user(filtered_df)
            if isinstance(avg_response_readable, dict):
                for user, time_str in avg_response_readable.items():
                    st.write(f"Average response time for **{user}**: {time_str}")
                if fastest_replier:
                    st.success(f"**{fastest_replier}** is the fastest replier!")
            else:
                st.info(avg_response_readable)

            st.subheader("Love Score (AI Analysis)")
            if st.button("Calculate Love Score"):
                with st.spinner("Calculating love score... This may take a moment."):
                    status, score, explanation, recommendations = calculate_love_score(filtered_df.copy())
                    if status == "Success" and score is not None:
                        st.metric(label="Love Score", value=f"{score}/100")
                        st.subheader("Explanation")
                        st.write(explanation)
                        if recommendations and recommendations != "N/A":
                            st.subheader("Recommendations")
                            st.write(recommendations)
                    else:
                        st.error(status)
            
            st.subheader("Top Romantic Messages")
            romantic_msgs = top_romantic_messages(filtered_df)
            if not romantic_msgs.empty:
                for idx, row in romantic_msgs.iterrows():
                    st.success(f"**{row['user']}** ({row['datetime'].strftime('%Y-%m-%d %H:%M')}): \"{row['message']}\" (Polarity: {row['polarity']:.2f})")
            else:
                st.info("No significantly romantic messages found in the filtered data. Try adjusting filters or uploading a chat with more positive sentiment.")

            st.subheader("Personality Traits (AI Guess)")
            if st.button("Guess Personality Traits"):
                with st.spinner("Analyzing personalities..."):
                    personality_output = guess_personality_traits(filtered_df.copy())
                    st.markdown(personality_output)

            st.subheader("Conflict & Recovery Patterns (AI Analysis)")
            if st.button("Detect Conflict Patterns"):
                with st.spinner("Analyzing conflict patterns..."):
                    conflict_analysis = detect_conflict_and_recovery(filtered_df.copy())
                    st.write(conflict_analysis)

            st.subheader("Relationship Compatibility Radar Chart")
            fig_radar, radar_error = create_compatibility_radar_chart(filtered_df.copy())
            if fig_radar:
                st.pyplot(fig_radar)
            else:
                st.info(radar_error)

        else:
            st.info("Please ensure your chat has exactly two participants to unlock full relationship insights.")


    else:
        st.info("Upload a WhatsApp chat file to get started!")

# Main app logic
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'

if st.session_state['logged_in']:
    show_main_app()
else:
    show_login_page()
