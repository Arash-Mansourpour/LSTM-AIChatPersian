import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
import logging
import json
import subprocess
import redis
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from transformers import pipeline, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer

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
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.tokenizer(x)
        out, _ = self.lstm(x)
        attn_weights = self.activation(self.attention(out))
        out = out * attn_weights
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out

# Initialize model parameters
input_size = 10
hidden_size = 20
output_size = 5
vocab_size = 1000
embedding_dim = 50
model = EnhancedAttentionLSTMModel(input_size, hidden_size, output_size, vocab_size, embedding_dim)

# Adam optimizer with L2 regularization
optimizers = {
    'adam': optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5),
    'rmsprop': optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-5)
}

# Warmup and Step learning rate scheduler
total_steps = 10000
scheduler = get_linear_schedule_with_warmup(optimizers['adam'], num_warmup_steps=500, num_training_steps=total_steps)

# Loss function
criterion = nn.CrossEntropyLoss()

# BLEU and ROUGE Evaluation
def evaluate_response(reference, candidate):
    # Calculate BLEU Score
    bleu_score = sentence_bleu([reference.split()], candidate.split())

    # Calculate ROUGE Score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference, avg=True)

    return {'bleu': bleu_score, 'rouge': rouge_scores}

# Training function with gradient clipping
def train(model, data_loader, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer = optimizers['adam']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        scheduler.step()

# Generate response using llama3.1
def generate_llama_response(context):
    process = subprocess.Popen(
        ["powershell", "-Command", "ollama run llama3.1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(input=context.encode())
    response = stdout.decode().strip()
    return response or "نتیجه‌ای یافت نشد."

# Google search function
def google_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"
    try:
        response = requests.get(url)
        data = response.json()
        results = data.get("items", [])
        return results[0]["snippet"] if results else "نتیجه‌ای یافت نشد."
    except requests.RequestException as e:
        logger.error("خطا در ارتباط با گوگل API: %s", e)
        return "مشکلی در انجام جستجو رخ داد."

# Feedback handler for model updates
async def feedback_handler(update: Update, context: CallbackContext):
    user_feedback = update.message.text
    user_message = context.user_data.get('pending_feedback')

    if user_message:
        if user_feedback.lower() == 'بله':
            logger.info("User confirmed the response as correct.")
        elif user_feedback.lower() == 'خیر':
            logger.info("User indicated the response was incorrect. Consider retraining the model.")

    context.user_data['pending_feedback'] = None

# Handling user messages and generating response
async def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text

    redis_client.lpush(f"chat_history:{update.message.chat_id}", user_message)
    chat_history = redis_client.lrange(f"chat_history:{update.message.chat_id}", 0, -1)
    chat_history = [msg.decode('utf-8') for msg in reversed(chat_history)]
    context_for_model = "\n".join(chat_history)

    llama_response = generate_llama_response(context_for_model)

    # Evaluate response quality
    eval_metrics = evaluate_response(user_message, llama_response)
    logger.info("Evaluation Metrics - BLEU: %s, ROUGE: %s", eval_metrics['bleu'], eval_metrics['rouge'])

    await update.message.reply_text(llama_response)
    await update.message.reply_text("آیا پاسخ درست بود؟ (بله/خیر)")
    context.user_data['pending_feedback'] = user_message

# Main function to start the bot
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.TEXT, feedback_handler))
    app.run_polling()

if __name__ == '__main__':
    main()
