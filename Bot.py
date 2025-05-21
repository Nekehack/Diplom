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
            WHERE user_id = '{message.chat.id}'
        """
    print(message.chat.id)
    cursor.execute(query)
    result = cursor.fetchone()
    role = result[0] if result else None

    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton("Анализ изображения", callback_data='analyze_image')
    print("И тут")
    print(role)
    if role == 'user':

        button2 = types.InlineKeyboardButton("Тех.поддержка", callback_data='tech_help')
        button3 = types.InlineKeyboardButton("Информация про бота", callback_data='info')
        button4 = types.InlineKeyboardButton("Профиль", callback_data='profil')
        button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
        markup.add(button1, button2, button)
        bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)

    # Добавляем кнопку "Другая кнопка" только для админов
    elif role == 'admin':
        # добавить подключение к другим базам данных, на роль админ

        # посмотреть список пользователей
        # посмотреть активность пользователей
        # удалить, добавить пользователя
        # изменить роль пользователю
        # смотреть настроки бота у пользователей
        # посмотреть отзывы
        # посмотреть часто задаваемые вопросы
        # посмотреть диалог пользователей с ботом

        button2 = types.InlineKeyboardButton("Данные пользователей", callback_data='bd_read')
        button3 = types.InlineKeyboardButton("Активность пользователей", callback_data='activity')
        button4 = types.InlineKeyboardButton("Добавить пользователя", callback_data='add')
        button5 = types.InlineKeyboardButton("Изменить роль пользователя", callback_data='user_edit')
        button6 = types.InlineKeyboardButton("Смотерть настройки бота у польщователя", callback_data='user_bot')
        button7 = types.InlineKeyboardButton("Отзывы", callback_data='review')
        button8 = types.InlineKeyboardButton("Вопросы", callback_data='question')
        button9 = types.InlineKeyboardButton("Диалоги", callback_data='dialog')
        button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

        markup.add(button1, button2,button3,
                   button4,button5,button6,
                   button7, button8,button9,
                   button)
        bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)
        print('Да, ты админ')

    elif role =='developer':

        # посмотреть активность пользователей
        # посмотреть настроки бота у пользователей
        # посмотреть предсказания модели у пуользователей
        # посмотреть структуру основной модели
        # посмотреть диалог с ботов

        button2 = types.InlineKeyboardButton("Активность", callback_data='activity')
        button3 = types.InlineKeyboardButton("Настроки бота", callback_data='bot_settings')
        button4 = types.InlineKeyboardButton("Предсказания ботов", callback_data='bot_predict')
        button5 = types.InlineKeyboardButton("Стуктура модели", callback_data='model_struct')
        button6 = types.InlineKeyboardButton("Диалоги", callback_data='dialog')
        button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

        markup.add(button1, button2, button3,
                   button4, button5, button6,
                   button)

        bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)

    else:
        markup.add(button1)
        bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)

    conn.commit()
    conn.close()

#Вызов необходимых функций
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):

    # Вызов функции анализа изображений
    if call.data == 'analyze_image':
        analis(call)

# Внешние функции для уровней пользователей

    #переходы по функциям обычного пользователя
    if call.data == 'tech_help':
        tech_help_user(call)

    elif call.data == 'info':
        bot_info(call)

    elif call.data == 'profil':
        user_profil(call)

    elif call.data == 'exit':
        work(call.message)

    # переходы по функциям админа
    elif call.data == 'bd_read':
        info_user_bd(call)

    elif call.data == 'activity':
        activity(call)

    elif call.data == 'add':
        add_user(call)

    elif call.data == 'user_edit':
        user_edit(call)

    elif call.data == 'user_bot':
        bot_user_setting(call)

    elif call.data == 'review':
        review(call)

    elif call.data == 'question':
        question(call)

    elif call.data == 'dialog':
        dialogs(call)

    elif call.data == 'exit':
        work(call.message)

    # переходы по функциям разработчика
    elif call.data == 'bot_settings':
        bot_settings(call)

    elif call.data == 'bot_predict':
        bot_predict(call)

    elif call.data == 'model_struct':
        model_struct(call)

    elif call.data == 'exit':
        work(call.message)


# Глубинные функции уровней пользователей



#ВЫПОЛНЕНИЕ ВНЕШНИХ ФУНКЦИИ ПО УРОВНЯМ
#Общая функция для анализа снимков
def analis(call):
    bot.send_message(call.message.chat.id, 'Вот твоё изображение')
    # сделать вызов загрузки нейронной сети
    print('Функция работает')
    work(call.message)


#функции обычного пользователя
def tech_help_user(call):
    print('Техподдержка работает')

def bot_info(call):
    print('bot info work')

def user_profil(call):
    print('user profil work')


#функции админа
def info_user_bd(call):
    print('bd read work')

def activity(call):
    print('activity')

def add_user(call):
    print('add user')

def user_edit(call):
    print('user edit')

def bot_user_setting(call):
    print('ueser settings bot')

def review(call):
    print('review')

def question(call):
    print('quastion')

def dialogs(call):
    print('dialog')

#функции разработчика
def bot_settings(call):
    print('bot settings')

def bot_predict(call):
    print('predict')

def model_struct(call):
    print('ыекгс')




def user(call):
    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton("Выход", callback_data='tzkjsdh')
    markup.add(button1)
    bot.send_message(call.message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)

    if call.data == 'tzkjsdh':
        print("Привет")

#функции для админа
def admin(call):
    bot.send_message(call.message.chat.id, 'Вот, что я могу для тебя сделать')
    work(call.message)

    #посмотреть список пользователей
    #посмотреть активность пользователей
    #удалить, добавить пользователя
    #изменить роль пользователю
    #смотреть настроки бота у пользователей
    # посмотреть отзывы
    # посмотреть часто задаваемые вопросы
    # посмотреть диалог пользователей с ботом

    pass


#функции для разработчика
def developer(message):
    bot.send_message(message.chat.id, 'Доступ к какой базе данных нужен?')
    work(message)

    # user_states[message.chat.id] = PASSWORD

    # посмотреть активность пользователей
    # посмотреть настроки бота у пользователей
    # посмотреть предсказания модели у пуользователей
    # посмотреть структуру основной модели
    # посмотреть диалог с ботов

bot.infinity_polling()
