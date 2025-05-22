import telebot
from telebot import types
import torchvision
import torch
from torchvision import transforms

from PIL import Image
from io import BytesIO

import pandas as pd
import os
import sqlite3

#токен для подключения к боту
token = '7433058915:AAHj5KtDTJ58OoGGUIayfWpDGOG3v0DnfgY'

#использование токена
bot = telebot.TeleBot(token)

#Словарь и константы для сохранения состояний
user_states = {}
WAITING_MENU, START, LOGIN, PASSWORD,PHOTO, WAITING_TEXT, DEV_BASE, WAITING_ID, WAITING_ANSWER, EDIT_LOGIN, EDIT_PASSWORD, ADMIN_ADD_USER = range(12)


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
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == WAITING_MENU)
def work(message):
    user_states[message.chat.id] = WAITING_MENU
    if user_states[message.chat.id] == WAITING_MENU:
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

        #кнопки пользователя
        if role == 'user':
            button2 = types.InlineKeyboardButton("Тех.поддержка", callback_data='tech_help')
            button3 = types.InlineKeyboardButton("Информация про бота", callback_data='info')
            button4 = types.InlineKeyboardButton("Профиль", callback_data='profil')
            markup.add(button1, button2, button3, button4)
            bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)

        #копки админа
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

            markup.add(button1, button2,button3,
                       button4,button5,button6,
                       button7, button8,button9)
            bot.send_message(message.chat.id, text="Вот доступные тебе функции", reply_markup=markup)
            print('Да, ты админ')

        # кнопки разработчика
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
            button7 = types.InlineKeyboardButton("Тех.поддержка", callback_data='help_tech')
            button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

            markup.add(button1, button2, button3,
                       button4, button5, button6,
                       button7)

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
        user_states[call.message.chat.id] = WAITING_TEXT
        bot.send_message(call.message.chat.id, 'Пожалуйста, введите описание вашей проблемы.')


    elif call.data == 'info':
        bot_info(call)

    elif call.data == 'profil':
        user_profil(call)

    # elif call.data == 'exit':
    #     work(call.message)

    # переходы по функциям админа
    elif call.data == 'bd_read':
        info_user_bd(call)

    elif call.data == 'activity':
        activity(call)

    elif call.data == 'add':
        # add_user(call)
        user_states[call.message.chat.id] = ADMIN_ADD_USER
        bot.send_message(call.message.chat.id, text='Введите id нового пользователя, имя, '
                                                    'его лоигн и пароль, а так же его роль')


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

    # elif call.data == 'exit':
    #     work(call.message)

    # переходы по функциям разработчика
    elif call.data == 'bot_settings':
        bot_settings(call)

    elif call.data == 'bot_predict':
        bot_predict(call)

    elif call.data == 'model_struct':
        model_struct(call)

    elif call.data == 'help_tech':
        # user_states[call.message.chat.id] = DEV_BASE
        # bot.send_message(call.message.chat.id, 'Вот список запросов от пользователей')
        help_tech_developer(call)


    elif call.data == 'exit':
        work(call.message)


# Глубинные функции уровней пользователей

    #глубокие функции пользователя
    elif call.data == 'edit_login':
        user_states[call.message.chat.id] = EDIT_LOGIN
        bot.send_message(call.message.chat.id, 'Введите новый логин')
        # edit_login(call)

    elif call.data == 'edit_password':
        user_states[call.message.chat.id] = EDIT_PASSWORD
        bot.send_message(call.message.chat.id, text='Введите новый пароль')
        # edit_password(call)

    #глубинные функции админа
    #взятие файлов из таблицы users
    elif call.data == 'admin_users_csv':
        send_amind_users_csv(call)

    elif call.data == 'admin_users_xlsx':
        send_amind_users_xlsx(call)


#ВЫПОЛНЕНИЕ ВНЕШНИХ ФУНКЦИИ ПО УРОВНЯМ
#Общая функция для анализа снимков
def analis(call):
    bot.send_message(call.message.chat.id, 'Вот твоё изображение')
    # сделать вызов загрузки нейронной сети
    print('Функция работает')
    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('Загрузить изображение', callback_data='analyze_image')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

    #повторый вызов функции или выход в меню
    markup.add(button1, button)
    bot.send_message(call.message.chat.id, text='Хочешь загрузить сообщение или выйти в меню?', reply_markup=markup)


    # work(call.message)



#функции обычного пользователя
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == WAITING_TEXT)
def tech_help_user(message):
    text = message.text
    print(f"Получено сообщение: {text}")
    print('Техподдержка работает')

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    print(message.chat.id)
    print(text)

    print(type(message.chat.id))
    print(type(text))

    query = "INSERT INTO help_history (user_id, request) VALUES (?, ?)"
    data = (message.chat.id, text)

    cursor.execute(query, data)

    bot.send_message(message.chat.id, text='Ваш запрос отправлен в техническую поддержку, ожидайте ответа.')
    # сбрасываем состояние
    # user_states[call.message.chat.id] = None
    user_states[message.chat.id] = WAITING_MENU
    work(message)

    conn.commit()
    conn.close()

#выдаёт информацию о боте
def bot_info(call):
    bot.send_message(call.message.chat.id, text='Вот информация обо мне:')
    bot.send_message(call.message.chat.id, text='Я медецинский Telegram-бот написанный для диплома')
    bot.send_message(call.message.chat.id, text='Основная задача нейронных '
                                                'сейтей, к котоырм я подклюёчн искать опухоли на снимках МРТ мозга человека'
                                                'для помощи врачам в их обноружении')
    bot.send_message(call.message.chat.id, text='Моя основная задача это помогать обычным пользователям '
                                                'самим посмотреть свои снимки')
    bot.send_message(call.message.chat.id, text='Ты так же можешь оставить комментарий по поводу работы со мной'
                                                ' или задать вопросы технической поддержке, если у тебя что-то случилось')
    bot.send_message(call.message.chat.id, text='В целом, это всё! Удачи.')

    work(call.message)

def user_profil(call):
    print('user profil work')

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = f"SELECT * FROM users WHERE user_id = '{call.message.chat.id}'"
    cursor.execute(query)
    result = cursor.fetchone()
    print(result)

    bot.send_message(call.message.chat.id, text=f'Ваше id: {result[1]}\n'
                                                f'Ваше имя: {result[2]}\n'
                                                f'Ваш логин: {result[3]}\n'
                                                f'Ваш пароль: {result[4]}\n')

    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('Сменить логин', callback_data='edit_login')
    button2 = types.InlineKeyboardButton('Сменить пароль', callback_data='edit_password')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')



    # повторый вызов функции или выход в меню
    markup.add(button1, button2, button)
    bot.send_message(call.message.chat.id, text='Изменить параметры или выйти?', reply_markup=markup)

    conn.commit()
    conn.close()

#глубокая функция по смене логина
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == EDIT_LOGIN)
def edit_login(message):
    text = message.chat.id
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = "UPDATE users SET login=? WHERE user_id = ?"
    data = (text, message.chat.id)
    cursor.execute(query, data)

    query = f"SELECT user_name FROM users WHERE user_id={message.chat.id}"
    cursor.execute(query)
    result = cursor.fetchone()

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

    # выход в меню
    markup.add(button)
    bot.send_message(message.chat.id, text=f'{result[0]} ваш логин изменён', reply_markup=markup)

#глубокая функция по смене пароля
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == EDIT_PASSWORD)
def edit_password(message):
    text = message.chat.id
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = "UPDATE users SET password=? WHERE user_id=?"
    data = (text, message.chat.id)
    cursor.execute(query, data)
    # cursor.execute("UPDATE users SET role = ? WHERE user_id = ? AND user_name = ?",
    #                ('user', '815722883', 'Илья'))

    query = f"SELECT user_name FROM users WHERE user_id={message.chat.id}"
    cursor.execute(query)
    result = cursor.fetchone()

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()

    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

    #выход в меню
    markup.add(button)
    bot.send_message(message.chat.id, text=f'{result[0]} ваш пароль изменён', reply_markup=markup)



#функции админа
#функция для отправки файлов из таблицы users
def info_user_bd(call):
    print('bd read work')

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()
    # df = pd.read_sql_query("SELECT * FROM users", conn)
    # df.to_csv('/Bot/bd_users.csv', index=False)
    # df.to_excel('/Bot/bd_users.xlsx', index=False, engine='openpyxl')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('CSV', callback_data='admin_users_csv')
    button2 = types.InlineKeyboardButton('XLSX', callback_data='admin_users_xlsx')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

    markup.add(button1, button2, button)
    bot.send_message(call.message.chat.id, text='В каком формате предоставить данные?', reply_markup=markup)

#отправка файла с данными пользователей в csv
def send_amind_users_csv(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM users", conn)
    df.to_csv('bd_users.csv', index=False)

    # Открываем файл и отправляем его пользователю
    with open('bd_users.csv', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список пользователей")
        os.remove('bd_users.csv')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)

#отправка файла с данными пользователей в xlsx
def send_amind_users_xlsx(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM users", conn)
    df.to_excel('bd_users.xlsx', index=False, engine='openpyxl')

    # Открываем файл и отправляем его пользователю
    with open('bd_users.xlsx', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список пользователей")
        os.remove('bd_users.xlsx')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)



def activity(call):
    print('activity')

@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == ADMIN_ADD_USER)
def add_user(message):
    print('add user')
    text = message.text
    text=text.split(':')

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()



    conn.commit()
    conn.close()

    print(text)






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


#функция обработки запроса пользователя
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == DEV_BASE)
def help_tech_developer(call):
    print('Function work')

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = "SELECT * FROM help_history"
    cursor.execute(query)
    rows = cursor.fetchall()
    for row in rows:
        user_id, user_problem = row[1], row[2]
        query = f"SELECT user_name FROM users WHERE user_id='{user_id}'"
        cursor.execute(query)
        result = cursor.fetchone()
        user_name = result[0]

        bot.send_message(call.message.chat.id, text=f'Пользователь {user_name} c id: {user_id} обратился с такой проблемой: "{user_problem}"')
    # user_states[call.message.chat.id] =



    conn.commit()
    conn.close()



    user_states[call.message.chat.id] = WAITING_ID
    bot.send_message(call.message.chat.id, text='Введи id пользователя и ответ')

    # work(call.message)

#функция ответа техподдержки
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == WAITING_ID)
def write_user_id(message):
    #классический список, первый ушёл, первый ушёл
    text = message.text
    text = text.split(':')

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = f'SELECT * FROM help_history WHERE user_id = {text[0]}'
    cursor.execute(query)
    result = cursor.fetchone()
    if result:
        cursor.execute(f'INSERT INTO help_history (assistant_id, assistant_answer) VAlUES (?, ?)', (message.chat.id, text[1]))
        bot.send_message(text[0], text=f'Здарвствуйте, ответ тех. поддержки, на ваш запрос: {text[1]}')

bot.infinity_polling()
