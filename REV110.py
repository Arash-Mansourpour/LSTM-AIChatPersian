import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, CallbackContext
import logging
import redis
import schedule
import time
import subprocess
import requests
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
from typing import List

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Redis Client for Persistent Chat History
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Tokenization and Embedding Layer
class Tokenizer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Tokenizer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Enhanced LSTM Model with Attention, Regularization, and Additional Linear Layers
class EnhancedAttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, embedding_dim):
        super(EnhancedAttentionLSTMModel, self).__init__()
        self.tokenizer = Tokenizer(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.tokenizer(x)
        out, _ = self.lstm(x)
        attn_weights = self.activation(self.attention(out))
        out = out * attn_weights
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        return out

# Initialize model parameters
input_size = 10
hidden_size = 20
output_size = 5
vocab_size = 1000
embedding_dim = 50
model = EnhancedAttentionLSTMModel(input_size, hidden_size, output_size, vocab_size, embedding_dim)

# Adam optimizer with L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000)

# Loss function
criterion = nn.CrossEntropyLoss()

# BLEU, ROUGE, and Perplexity Evaluation
def evaluate_response(reference, candidate):
    bleu_score = sentence_bleu([reference.split()], candidate.split())
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference, avg=True)
    return {'bleu': bleu_score, 'rouge': rouge_scores}

# Diverse Sampling for response generation
def diverse_sampling(contexts):
    possible_responses = [generate_llama_response(context) for context in contexts]
    best_response = max(possible_responses, key=lambda r: evaluate_response(contexts[0], r)['bleu'])
    return best_response

# Generate response using llama 3.1
def generate_llama_response(context):
    process = subprocess.Popen(
        ["powershell", "-Command", "ollama run llama3.1"],  # Ensure the correct command for llama 3.1
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
        llama_response = diverse_sampling(chat_history)
        eval_metrics = evaluate_response(user_message, llama_response)
        logger.info("Evaluation Metrics - BLEU: %s, ROUGE: %s", eval_metrics['bleu'], eval_metrics['rouge'])
        await update.message.reply_text(llama_response)

    await update.message.reply_text("آیا پاسخ درست بود؟ (بله/خیر)")
    context.user_data['pending_feedback'] = user_message

# Self-improvement mechanism
def self_improve_model(chat_history):
    for i in range(len(chat_history) - 1):
        reference = chat_history[i]
        candidate = chat_history[i + 1]
        eval_metrics = evaluate_response(reference, candidate)
        if eval_metrics['bleu'] < 0.5:  
            logger.info("پاسخ نیاز به بهبود دارد، بازآموزی مدل در حال انجام است.")

# Main function to start the bot
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.TEXT, feedback_handler))
    app.run_polling()

if __name__ == '__main__':
    main()
