import telebot
from telebot import types
import torchvision
import torch
from torchvision import transforms

from PIL import Image
from io import BytesIO

import sqlite3

#токен для подключения к боту
token = '74gY'

#использование токена
bot = telebot.TeleBot(token)

user_states = {}
START, LOGIN, PASSWORD, PHOTO = range(4)

@bot.message_handler(commands=['start'])
def start(message):
    user_states[message.chat.id] = START
    bot.send_message(message.chat.id,
                     text=f"Привет, {message.from_user.first_name}, введи свой логин и пароль, для авторизации!")
    bot.send_message(message.chat.id,text='login:')


    # markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    # markup.add('Анализ снимка МРТ', 'Тех.поддержка')
    # bot.send_message(
    #     message.chat.id,
    #     f"Привет, {message.from_user.first_name}! Выбери команду.",
    #     reply_markup=markup
    # )
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == START)
def login(message):
    user_states[message.chat.id] = LOGIN
    user_states[f'{message.chat.id}_login'] = message.text
    bot.send_message(message.chat.id, text='password:')


@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == LOGIN)
def password(message):
    user_states[message.chat.id] = PASSWORD
    password = message.text
    user_states[f'{message.chat.id}_password'] = message.text


    if user_states[f'{message.chat.id}_login'] == '1' and user_states[f'{message.chat.id}_password'] == '1':
        # bot.send_message(message.chat.id, text=f'Отлично твой логин: '
        #                                        f'{user_states[f"{message.chat.id}_login"]} '
        #                                        f'и пароль {user_states[f"{message.chat.id}_password"]}')
        role = 'user'
        print(message.from_user.id)
        print(message.from_user.first_name)
        print(user_states[f'{message.chat.id}_login'])
        print(user_states[f'{message.chat.id}_password'])
        print(role)
        conn = sqlite3.connect('bot_base.db')
        cursor = conn.cursor()

        query = """
        SELECT * FROM users
        WHERE user_id=815722883
          AND user_name='Илья'
          AND login='1'
          AND password='1'
          AND role='user'
        """

        if cursor.execute(query):
            bot.send_message(message.chat.id, text='Отлично, ты уже есть в базе!')
        else:
            cursor.execute("INSERT INTO users (user_id, user_name, login, password, role) VAlUES (?, ?, ?, ?, ?)",
                           (message.from_user.id,
                            message.from_user.first_name,
                            user_states[f'{message.chat.id}_login'],
                            user_states[f'{message.chat.id}_password'],
                            role))
        work(message)
    else:
        bot.send_message(message.chat.id, text='Твой логин или пароль неправельный, попробуй ещё раз')
        user_states[message.chat.id] = None
        start(message)

@bot.message_handler(content_types=['text'])
def work(message):
    pass


# @bot.message_handler(content_types=['text'])
# def

# @bot.message_handler(func=lambda m: True)
# def handle_message(message):
#     user_id = message.from_user.id
#     text = message.text
#
#     if text == 'Анализ снимка МРТ':
#         user_states[user_id] = 'waiting_for_image'
#         bot.send_message(message.chat.id, "Пожалуйста, отправьте изображение МРТ для анализа.")
#         return
#     elif text == 'Тех.поддержка':
#         bot.send_message(message.chat.id, "Техподдержка в разработке.")
#         return
#     else:
#         # Если ожидаем фото, но пришел текст
#         if user_states.get(user_id) == 'waiting_for_image':
#             bot.send_message(message.chat.id, "Пожалуйста, отправьте изображение для анализа.")
#         else:
#             bot.send_message(message.chat.id, "Я вас понял. Выберите команду.")
#
# # Обработчик для фото
# @bot.message_handler(content_types=['photo'])
# def handle_photo(message):
#     user_id = message.from_user.id
#     if user_states.get(user_id) == 'waiting_for_image':
#         try:
#             # Получаем файл фото
#             file_id = message.photo[-1].file_id
#             file_info = bot.get_file(file_id)
#             downloaded_file = bot.download_file(file_info.file_path)
#
#             # Создаем BytesIO объект
#             image_stream = BytesIO(downloaded_file)
#
#             # Анализируем изображение
#             bot.send_message(message.chat.id, "Обрабатываю изображение, пожалуйста, подождите...")
#             mrt_analis(image_stream, message)
#
#             # После обработки сбрасываем состояние
#             user_states[user_id] = None
#         except Exception as e:
#             bot.send_message(message.chat.id, f"Ошибка при обработке изображения: {e}")
#     else:
#         bot.send_message(message.chat.id, "Пожалуйста, сначала выберите команду для анализа.")
#
# def mrt_analis(image_stream, message):
#     model_path = '/Users/ilia/DeiT/DeiT_weights.pth'
#     # Загружаем модель
#     try:
#         # Используйте вашу собственную реализацию DeiT
#         model = DeiT(
#             in_channels=3,
#             patch_size=16,
#             emb_size=384,  # важно совпадение с тем, что было при обучении
#             img_size=224,
#             depth=12,
#             n_classes=2
#         )
#         model.load_state_dict(torch.load(model_path, map_location='cpu'))
#         model.eval()
#
#
#     except Exception as e:
#         # bot.send_message(message.chat.id, f"Ошибка загрузки модели: {e}")
#         print(e)
#         return
#
#     # Трансформы
#     transform = transforms.Compose([
#         transforms.Resize((150, 150)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])
#
#     # Предсказание
#     def predict_image(image_stream):
#         image = Image.open(image_stream).convert('RGB')
#         image = transform(image)
#         image = image.unsqueeze(0)
#         with torch.no_grad():
#             output = model(image)
#         return output
#
#     predictions = predict_image(image_stream)
#     # Тут можно обработать predictions и отправить результат
#     bot.send_message(message.chat.id, f"Результат анализа: {str(predictions)}")
#     print(predictions)
#     print(1)

bot.infinity_polling()
