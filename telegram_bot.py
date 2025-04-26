import telebot
import os
from dotenv import load_dotenv
from modules.update_predict_mines import update_model_and_predict
from modules.logger import log_info

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start_handler(message):
    bot.send_message(message.chat.id, "Assalomu alaykum! Mines AI signal botiga xush kelibsiz.")

@bot.message_handler(commands=['signal'])
def signal_handler(message):
    result, error = update_model_and_predict()
    if error:
        bot.send_message(message.chat.id, f"Xatolik: {error}")
    else:
        bot.send_message(message.chat.id, f"Eng xavfsiz kataklar: {', '.join(result)}")
        with open("data/chart.png", "rb") as photo:
            bot.send_photo(message.chat.id, photo)

bot.polling()
