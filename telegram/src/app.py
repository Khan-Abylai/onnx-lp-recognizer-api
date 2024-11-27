import os
import cv2
import logging
import requests
from config import TOKEN, URL
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import shortuuid

BASE_DIR = os.getcwd()

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="Я бот для распознавания номерных знаков. Отправьте фото с номером, и я обработаю его.")


def unknown(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Извините, я не понимаю эту команду.")


def callback_fromImage_toCaption(update, context):
    try:
        img_file = context.bot.get_file(update.message.photo[-1].file_id)
        filename = os.path.join('static', 'tmp', shortuuid.uuid() + '.png')
        img_file.download(filename)
        logging.info(f"Изображение сохранено: {filename}")

        img = cv2.imread(filename)
        if img is None:
            raise ValueError("Не удалось загрузить изображение.")

        file = cv2.imencode(".png", img)[1].tobytes()
        files = {"file": ("photo.png", file, "image/png")}
        response = requests.post(URL, files=files)

        if response.status_code == 200 and response.json().get("status"):
            label = response.json()["data"]["label"]
            confidence = response.json()["data"]["confidence"]
            context.bot.send_message(chat_id=update.effective_chat.id,
                                     text=f"Распознавание: {label.upper()}, вероятность: {confidence:.4f}")
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text="Номер не найден.")
            logging.info(f"Ошибка API: {response.text}")

    except Exception as e:
        logging.error(f"Ошибка обработки изображения: {str(e)}")
        context.bot.send_message(chat_id=update.effective_chat.id, text="Произошла ошибка при обработке.")
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def main():
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.command, unknown))
    dispatcher.add_handler(MessageHandler(Filters.photo, callback_fromImage_toCaption))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
