import telebot
from telebot import types
import torchvision
import torch
from torchvision import transforms

from PIL import Image
from io import BytesIO

import sqlite3

#токен для подключения к боту
token = '7433058915:AAHj5KtDTJ58OoGGUIayfWpDGOG3v0DnfgY'

#использование токена
bot = telebot.TeleBot(token)

#Словарь и константы для сохранения состояний
user_states = {}
START, LOGIN, PASSWORD,PHOTO = range(4)

#Стартовая часть с проверкой логина и пароля
@bot.message_handler(commands=['start'])
def start(message):
    user_states[message.chat.id] = START
    bot.send_message(message.chat.id,
                     text=f"Привет, {message.from_user.first_name}, введи свой логин и пароль, для авторизации!")
    bot.send_message(message.chat.id,text='login:')

#Ввод логина
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == START)
def login(message):
    user_states[message.chat.id] = LOGIN
    user_states[f'{message.chat.id}_login'] = message.text
    bot.send_message(message.chat.id, text='password:')

#Вводы и проверка пароля
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == LOGIN)
def password(message):
    user_states[message.chat.id] = PASSWORD
    user_states[f'{message.chat.id}_password'] = message.text

    #Проверка
    #Незабыть добавить более корректную првоерку
    if user_states[f'{message.chat.id}_login'] == '1' and user_states[f'{message.chat.id}_password'] == '1':
        # bot.send_message(message.chat.id, text=f'Отлично твой логин: '
        #                                        f'{user_states[f"{message.chat.id}_login"]} '
        #                                        f'и пароль {user_states[f"{message.chat.id}_password"]}')

        #Тестовая часть удалить
        print(message.from_user.id)
        print(message.from_user.first_name)
        login = user_states[f'{message.chat.id}_login']
        password = user_states[f'{message.chat.id}_password']

        #Подключение к базе данных
        conn = sqlite3.connect('bot_base.db')
        cursor = conn.cursor()

        #Запрос
        query = f"""
        SELECT * FROM users
        WHERE user_id={message.from_user.id}
          AND user_name='{message.from_user.first_name}'
          AND login='{login}'
          AND password='{password}'
        """

        cursor.execute(query)
        result = cursor.fetchone()

        #Проверка на наличие в базе
        if result:
            bot.send_message(message.chat.id, text='Вход выолнен!')

            query = f"""
                        SELECT role FROM users
                        WHERE user_id = {message.chat.id}
                        AND user_name = '{message.from_user.first_name}'
                    """

            cursor.execute(query)
            result = cursor.fetchone()

            print(result)
            print(result[0] if result else None)
            role = result[0] if result else None

            bot.send_message(message.chat.id, text=f'Твоя текущая роль: {role}')

        else:
            cursor.execute("INSERT INTO users (user_id, user_name, login, password, role) VAlUES (?, ?, ?, ?, ?)",
                           (message.from_user.id,
                            message.from_user.first_name,
                            user_states[f'{message.chat.id}_login'],
                            user_states[f'{message.chat.id}_password'],
                            'user'))
        work(message)
        conn.commit()
        conn.close()

    #Если логин или пароль не совпали
    # else:
    #     bot.send_message(message.chat.id, text='Твой логин или пароль неправельный, попробуй ещё раз')
    #     user_states[message.chat.id] = None
    #     start(message)


# функции кнопок
@bot.message_handler(content_types=['text'])
def work(message):
    print('ЗДесь работает')
    # Создаем соединение с базой данных
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    # Выполняем запрос для определения роли пользователя
    query = f"""
            SELECT role FROM users
            WHERE user_id = {message.chat.id}
            AND user_name = '{message.from_user.first_name}'
        """

    cursor.execute(query)
    result = cursor.fetchone()
    role = result[0] if result else None

    markup = types.InlineKeyboardMarkup()
    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton("Анализ изображения", callback_data='analyze_image')
    print("И тут")
    if role == 'user':
        button2 = types.InlineKeyboardButton("Тех.поддержка", callback_data='tech_help')
        markup.add(button1, button2)
        bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)

    # Добавляем кнопку "Другая кнопка" только для админов
    elif role == 'admin':
        # добавить подключение к другим базам данных, на роль админ
        button2 = types.InlineKeyboardButton("Другая кнопка", callback_data='other_action')
        markup.add(button1, button2)
        bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)
        print('Да, ты админ')

    elif role =='developer':
        button2 = types.InlineKeyboardButton("Что хочешь узнать?", callback_data='what_need')
        markup.add(button1, button2)
        bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)

    else:
        markup.add(button1)
        bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)

    conn.commit()
    conn.close()

#Вызов необходимых функций
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    #вызов функции анализ изображения
    if call.data == 'analyze_image':
        analis(call.message)
    # вызов функции тех поддержки
    elif call.data == 'tech_help':
        tech(call.message)
    # вызов функции другое
    elif call.data == 'other_action':
        admin(call.message)

    # вызов функции для разработчика
    elif call.data == 'what_need':
        developer(call.message)

def analis(message):
    bot.send_message(message.chat.id, 'Вот твоё изображение')
    # После выполнения возвращаем в меню
    work(message)

def tech(message):
    bot.send_message(message.chat.id, 'Обращение в тех поддержку')
    # После выполнения возвращаем в меню
    work(message)

def admin(message):
    bot.send_message(message.chat.id, 'Вот, что я могу для тебя сделать')
    work(message)

def developer(message):
    bot.send_message(message.chat.id, 'Доступ к каой базе данных нужен?')
    work(message)

    # user_states[message.chat.id] = PASSWORD

bot.infinity_polling()
