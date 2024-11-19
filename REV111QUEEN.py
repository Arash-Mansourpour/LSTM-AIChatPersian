import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from dotenv import load_dotenv
import logging
import redis
from telegram import Update
import requests
from telegram.ext import ApplicationBuilder, MessageHandler, filters, CallbackContext
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import subprocess
from typing import List
import threading

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Redis Client for Persistent Chat History
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Enhanced Transformer with Self-Attention and Residual Connections
class EnhancedTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, embedding_dim, num_heads=8, dropout=0.3):
        super(EnhancedTransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        attn_output, _ = self.self_attention(x, x, x)
        attn_output = self.layer_norm(attn_output + x)  # Residual Connection
        out = self.dropout(attn_output[:, -1, :])
        out = self.activation(self.fc1(out))
        out = self.fc2(out)
        return out

# Initialize model parameters
input_size = 10
hidden_size = 128
output_size = 5
vocab_size = 1000
embedding_dim = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedTransformerModel(input_size, hidden_size, output_size, vocab_size, embedding_dim).to(device)

# Adam optimizer with L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000)

# Loss function
criterion = nn.CrossEntropyLoss()

# BLEU and ROUGE Evaluation
def evaluate_response(reference, candidate):
    bleu_score = sentence_bleu([reference.split()], candidate.split())
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference, avg=True)
    return {'bleu': bleu_score, 'rouge': rouge_scores}

# Diverse Sampling for response generation
def diverse_sampling(contexts):
    possible_responses = [generate_qwen_response(context) for context in contexts]
    best_response = max(possible_responses, key=lambda r: evaluate_response(contexts[-1], r)['bleu'])
    return best_response

# Generate response using qwen2.5-coder:14b
def generate_qwen_response(context):
    process = subprocess.Popen(
        ["powershell", "-Command", "ollama run qwen2.5-coder:14b"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate(input=context.encode())
    return stdout.decode().strip() or "نتیجه‌ای یافت نشد."

# Google Search for real-time information
def google_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
    response = requests.get(url)
    results = response.json().get("items", [])
    return "\n".join([item["snippet"] for item in results[:3]]) if results else "نتیجه‌ای یافت نشد."

# Self-improvement mechanism
def self_improve_model(chat_history):
    for i in range(len(chat_history) - 1):
        reference = chat_history[i]
        candidate = chat_history[i + 1]
        eval_metrics = evaluate_response(reference, candidate)
        if eval_metrics['bleu'] < 0.5:
            logger.info("پاسخ نیاز به بهبود دارد، بازآموزی مدل در حال انجام است.")

# Periodic self-improvement every 3 minutes
def periodic_self_improvement():
    while True:
        all_chat_ids = redis_client.keys("chat_history:*")
        for chat_id in all_chat_ids:
            chat_history = redis_client.lrange(chat_id, 0, 10)
            chat_history = [msg.decode('utf-8') for msg in reversed(chat_history)]
            self_improve_model(chat_history)
        time.sleep(180)

# Start the periodic self-improvement in a separate thread
threading.Thread(target=periodic_self_improvement, daemon=True).start()

# Feedback handler for model updates
async def feedback_handler(update: Update, context: CallbackContext):
    user_feedback = update.message.text
    if context.user_data.get('pending_feedback'):
        if user_feedback.lower() == 'بله':
            logger.info("User confirmed the response as correct.")
        elif user_feedback.lower() == 'خیر':
            logger.info("User indicated the response was incorrect. Model will self-improve.")
            self_improve_model(redis_client.lrange(f"chat_history:{update.message.chat_id}", 0, 10))
        context.user_data['pending_feedback'] = None

# Handle user messages and generate response
async def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text
    redis_client.lpush(f"chat_history:{update.message.chat_id}", user_message)
    chat_history = redis_client.lrange(f"chat_history:{update.message.chat_id}", 0, 10)
    chat_history = [msg.decode('utf-8') for msg in reversed(chat_history)]

    if "جستجو" in user_message:
        query = user_message.replace("جستجو", "").strip()
        google_response = google_search(query)
        await update.message.reply_text(google_response)
    else:
        qwen_response = diverse_sampling(chat_history)
        eval_metrics = evaluate_response(user_message, qwen_response)
        logger.info("Evaluation Metrics - BLEU: %s, ROUGE: %s", eval_metrics['bleu'], eval_metrics['rouge'])
        await update.message.reply_text(qwen_response)

    await update.message.reply_text("آیا پاسخ درست بود؟ (بله/خیر)")
    context.user_data['pending_feedback'] = user_message

# Main function to start the bot
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.TEXT, feedback_handler))
    app.run_polling()

if __name__ == '__main__':
    main()
