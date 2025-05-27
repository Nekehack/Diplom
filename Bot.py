import telebot
from telebot import types

import os

from PIL import Image
from io import BytesIO

import pandas as pd

import sqlite3
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import cv2

#токен для подключения к боту
token = '7433058915:AAHj5KtDTJ58OoGGUIayfWpDGOG3v0DnfgY'

#использование токена
bot = telebot.TeleBot(token)

#Словарь и константы для сохранения состояний
user_states = {}
WAITING_MENU, START, LOGIN, PASSWORD, PHOTO, WAITING_TEXT, DEV_BASE, WAITING_ID, WAITING_ANSWER, EDIT_LOGIN, EDIT_PASSWORD, WAITING_REVIEW, ADMIN_ADD_USER, ADMIN_USER_EDIT, ADMIN_WRITE_USER, THC_PHOTO, TWC_PHOTO = range(17)
PHOTO_MRT = 15


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
            button5 = types.InlineKeyboardButton("Отзыв", callback_data='review')
            markup.add(button1, button2, button3, button4, button5)
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
            button6 = types.InlineKeyboardButton("Смотерть настройки бота", callback_data='bot_settings_admin')
            button7 = types.InlineKeyboardButton("Отзывы", callback_data='review_admin')
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

    #Вызов общей функции анализа изображений
    if call.data == 'analyze_image':
        analis(call)

    elif call.data == 'thc':
        user_states[call.message.chat.id] = THC_PHOTO
        bot.send_message(call.message.chat.id, text='Используется модель THCMax. Загрузите снимок')

    elif call.data == 'twc':
        user_states[call.message.chat.id] = TWC_PHOTO
        bot.send_message(call.message.chat.id, text='Используется упрощённаая модель TWCLow. Загрузите снимок')

# Внешние функции для уровней пользователей

    #переходы по функциям обычного пользователя
    elif call.data == 'tech_help':
        user_states[call.message.chat.id] = WAITING_TEXT
        bot.send_message(call.message.chat.id, 'Пожалуйста, введите описание вашей проблемы')


    elif call.data == 'info':
        bot_info(call)

    elif call.data == 'profil':
        user_profil(call)

    elif call.data == 'review':
        user_states[call.message.chat.id] = WAITING_REVIEW
        bot.send_message(call.message.chat.id, 'Напишите отзыв о работе бота, который хотите оставить')

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
        # user_edit(call)
        user_states[call.message.chat.id] = ADMIN_USER_EDIT
        bot.send_message(call.message.chat.id, text='Введите id пользователя, параметры которого хотите поменять:'
                                                    'id, его лоигн и пароль, а так же его роль')

    elif call.data == 'bot_settings_admin':
        bot_settings_admin(call)

    elif call.data == 'review_admin':
        review_admin(call)

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
    #смена логина и пароля
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

    #взятие файлов из таблицы настройки
    elif call.data == 'bot_settings_admin_csv':
        bot_settings_admin_csv(call)

    elif call.data == 'bot_settings_admin_xlsx':
        bot_settings_admin_xlsx(call)

    #взятие файлов из таблицы отзывы
    elif call.data == 'admin_review_csv':
        admin_review_csv(call)

    elif call.data == 'admin_review_xlsx':
        admin_review_csv(call)

    # взятие файлов из таблицы вопросы
    elif call.data == 'admin_question_csv':
        admin_question_csv(call)

    elif call.data == 'admin_question_xlsx':
        admin_question_xlsx(call)

    #отправка сообщения пользователю
    elif call.data == 'send_admin_message':
        user_states[call.message.chat.id] = ADMIN_WRITE_USER
        bot.send_message(call.message.chat.id, 'Введите сообщение для пользователя')

    #глубинный функции разработчика
    #взятие файлов с настройками бота

    elif call.data == 'bot_settings_developer_csv':
        bot_settings_developer_csv(call)

    elif call.data == 'bot_settings_developer_xlsx':
        bot_settings_developer_xlsx(call)

    #взятие файлов по предсказаниям моделей
    elif call.data == 'bot_predictions_developer_csv':
        bot_predictions_developer_csv(call)

    elif call.data == 'bot_predictions_developer_xlsx':
        bot_predictions_developer_xlsx(call)

#ВЫПОЛНЕНИЕ ВНЕШНИХ ФУНКЦИИ ПО УРОВНЯМ
#Общая функция для анализа снимков
def analis(call):
    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('Мощная нейронка', callback_data='thc')
    button2 = types.InlineKeyboardButton('Слабая нейронка', callback_data='twc')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button1, button2, button)

    bot.send_message(call.message.chat.id, text='Выберите модель', reply_markup=markup)

@bot.message_handler(content_types=['photo'], func=lambda m: user_states.get(m.chat.id) == THC_PHOTO)
def thc(message):
    print('sdkjfh`')
    photo_list = message.photo
    largest_photo = photo_list[-1]
    file_info = bot.get_file(largest_photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('mrt.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.send_message(message.chat.id, text="Фотография успешно сохранена!")

    #Загрузка основной модели:

    class HardDistillationLoss(nn.Module):
        def __init__(self, teacher: nn.Module):
            super().__init__()
            self.teacher = teacher
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, inputs: Tensor, outputs: tuple[Tensor, Tensor], labels: Tensor) -> Tensor:
            outputs_cls, outputs_dist = outputs

            # Базовая потеря (CLS)
            base_loss = self.criterion(outputs_cls, labels)

            # Вычисляем предсказания учителя
            with torch.no_grad():
                teacher_outputs = self.teacher(inputs)

            # Ограничиваем выходы учителя двумя классами
            teacher_logits = teacher_outputs[:, :2]  # Берем только первые два класса
            teacher_labels = torch.argmax(teacher_logits, dim=1)

            # Потеря для DIST
            teacher_loss = self.criterion(outputs_dist, teacher_labels)

            # Комбинируем потери
            return 0.5 * base_loss + 0.5 * teacher_loss

    class PatchEmbedding(nn.Module):
        def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
            super().__init__()
            self.patch_size = patch_size

            # Проекция патчей
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )

            # Токены CLS и DIST
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
            self.dist_token = nn.Parameter(torch.randn(1, 1, emb_size))  # Убедитесь, что это определено

            # Позиционные эмбеддинги
            num_patches = (img_size // patch_size) ** 2
            self.positions = nn.Parameter(torch.randn(num_patches + 2, emb_size))  # +2 для cls_token и dist_token

        def forward(self, x: Tensor) -> Tensor:
            b, _, _, _ = x.shape

            # Проекция патчей
            x = self.projection(x)

            # Создание токенов CLS и DIST
            cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
            dist_tokens = repeat(self.dist_token, '() n e -> b n e', b=b)

            # Добавление токенов CLS и DIST к входным данным
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)

            # Добавление позиционных эмбеддингов
            x += self.positions

            return x

    class ClassificationHead(nn.Module):
        def __init__(self, emb_size: int = 768, n_classes: int = 2):
            super().__init__()

            self.head = nn.Linear(emb_size, n_classes)
            self.dist_head = nn.Linear(emb_size, n_classes)

        def forward(self, x: Tensor) -> Tensor:
            x, x_dist = x[:, 0], x[:, 1]
            x_head = self.head(x)
            x_dist_head = self.dist_head(x_dist)

            if self.training:
                x = x_head, x_dist_head  # Возвращает кортеж
            else:
                x = (x_head + x_dist_head) / 2  # Возвращает тензор
            return x

    class MultiHeadAttention(nn.Module):
        def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
            super().__init__()
            self.emb_size = emb_size
            self.num_heads = num_heads
            # fuse the queries, keys and values in one matrix
            self.qkv = nn.Linear(emb_size, emb_size * 3)
            self.att_drop = nn.Dropout(dropout)
            self.projection = nn.Linear(emb_size, emb_size)

        def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
            # split keys, queries and values in num_heads
            qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
            queries, keys, values = qkv[0], qkv[1], qkv[2]
            # sum up over the last axis
            energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
            if mask is not None:
                fill_value = torch.finfo(torch.float32).min
                energy = energy.masked_fill(~mask, fill_value)

            scaling = self.emb_size ** 0.5
            att = F.softmax(energy / scaling, dim=-1)
            att = self.att_drop(att)
            # sum up over the third axis
            out = torch.einsum('bhqk, bhkd -> bhqd', att, values)
            out = rearrange(out, "b h n d -> b n (h d)")
            out = self.projection(out)
            return out

    class ResidualAdd(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x, **kwargs):
            res = x
            x = self.fn(x, **kwargs)
            x += res
            return x

    class FeedForwardBlock(nn.Sequential):
        def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
            super().__init__(
                nn.Linear(emb_size, expansion * emb_size),
                nn.GELU(),
                nn.Dropout(drop_p),
                nn.Linear(expansion * emb_size, emb_size),
            )

    class TransformerEncoderBlock(nn.Sequential):
        def __init__(self,
                     emb_size: int = 768,
                     drop_p: float = 0.,
                     forward_expansion: int = 4,
                     forward_drop_p: float = 0.,
                     **kwargs):
            super().__init__(
                ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                )),
                ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                )
                ))

    class TransformerEncoder(nn.Sequential):
        def __init__(self, depth: int = 12, **kwargs):
            super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

    class DeiT(nn.Sequential):
        def __init__(self,
                     in_channels: int = 3,
                     patch_size: int = 16,
                     emb_size: int = 768,
                     img_size: int = 224,
                     depth: int = 12,
                     n_classes: int = 1000,
                     **kwargs):
            super().__init__(
                PatchEmbedding(in_channels, patch_size, emb_size, img_size),
                TransformerEncoder(depth, emb_size=emb_size, **kwargs),
                ClassificationHead(emb_size, n_classes))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Изменяем размер до 224x224
        transforms.ToTensor(),  # Преобразуем в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
    ])

    # Создание датасета с помощью ImageFolder
    ds = datasets.ImageFolder(root='Testing', transform=transform)

    # Создание DataLoader
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    print(ds.classes)  # ['tumor', 'no_tumor']
    print(len(ds))

    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None

            # Hook для сохранения градиентов и активаций
            target_layer.register_forward_hook(self.save_activations)
            target_layer.register_backward_hook(self.save_gradients)

        def save_activations(self, module, input, output):
            self.activations = output.detach()

        def save_gradients(self, module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def forward(self, x, class_idx=None):
            # Сохраняем исходные размеры изображения
            original_size = x.shape[-2:]  # (height, width)
            h, w = original_size
            print('h: ', h, 'w: ', w)

            # Проверка, что размеры корректны
            if h <= 0 or w <= 0:
                raise ValueError(f"Некорректные размеры изображения: height={h}, width={w}")

            # Прямой проход через модель
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]  # Берём первый выход (CLS)
            self.model.zero_grad()

            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

            one_hot = torch.zeros_like(logits)
            one_hot[0][class_idx] = 1
            one_hot.requires_grad_(True)

            # Вычисляем градиенты относительно one_hot
            output = (one_hot * logits).sum()
            output.backward(retain_graph=True)

            gradients = self.gradients.cpu().numpy()[0]
            activations = self.activations.cpu().numpy()[0]

            weights = np.mean(gradients, axis=(1, 2))
            print('h: ', h, 'w: ', w)
            cam = np.zeros(activations.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                w = 224
                cam += w * activations[i]

            cam = np.maximum(cam, 0)
            print('h: ', h, 'w: ', w)

            # Проверка размеров перед изменением размера
            if int(w) <= 0 or int(h) <= 0:
                raise ValueError(f"Некорректные размеры для изменения размера: w={w}, h={h}")
            # w=224
            cam = cv2.resize(cam, (int(w), int(h)))  # Преобразуем w и h в целые числа

            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            return cam

        def __call__(self, x, class_idx=None):
            return self.forward(x, class_idx)

    def load_model_for_analysis(model_path: str, device: str = 'cpu'):
        """
        Загружает сохранённую модель PyTorch для анализа.
        :param model_path: путь к .pth файлу
        :param device: 'cpu' или 'cuda' (по умолчанию 'cpu')
        :return: модель
        """
        model = torch.load(model_path, map_location=torch.device(device))
        model.eval()
        return model

    # Пример использования:
    # model = load_model_for_analysis('DeiT.pth', device='cpu')

    def predict_image(model, image_path, transform, class_names, device='cpu'):
        """
        Делает предсказание для одного изображения с помощью загруженной модели.
        :param model: загруженная модель
        :param image_path: путь к изображению
        :param transform: torchvision.transforms для обработки изображения
        :param class_names: список имён классов (например, ds.classes)
        :param device: 'cpu' или 'cuda'
        :return: имя класса и вероятность
        """
        model.eval()
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)  # добавляем batch dimension

        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):  # если модель возвращает кортеж (как в DeiT)
                output = output[0]
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            pred_class = class_names[pred_idx]
            pred_prob = probs[0, pred_idx].item()
        return pred_class, pred_prob

    # Пример использования:
    # model = load_model_for_analysis('/Users/ilia/DeiT/DeiT.pth')
    # class_name, prob = predict_image(model, 'path/to/image.jpg', transform, ds.classes)
    # print(f'Класс: {class_name}, вероятность: {prob:.2f}')

    # 1. Загружаем модель
    model = load_model_for_analysis('/Users/ilia/DeiT/DeiT.pth', device='cpu')

    # 2. Делаем предсказание
    class_name, prob = predict_image(model, 'Testing/tumor/пример.jpg', transform, ds.classes)
    print(f'Класс: {class_name}, вероятность: {prob:.2f}')




    photo_path = 'mrt.jpg'
    with open('mrt.jpg', 'rb') as photo:
        bot.send_photo(message.chat.id, photo=photo)
    os.remove('mrt.jpg')





@bot.message_handler(content_types=['photo'], func=lambda m: user_states.get(m.chat.id) == TWC_PHOTO)
def twc(message):
    #загрузка и использование модели
    try:
        photo_list = message.photo
        largest_photo = photo_list[-1]
        file_info = bot.get_file(largest_photo.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open('mrt.jpg', 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.send_message(message.chat.id, text="Фотография успешно сохранена!")

        img_path = 'mrt.jpg'
        img = load_img(img_path, target_size=(150, 150))
        x = img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        # загружаем модель
        model = tf.keras.models.load_model('/Users/ilia/brainTumorClassification/my_model1.keras')

        # предсказание
        preds = model.predict(x)
        print(preds)

        #отправка файла с результатами в чат
        with open('mrt.jpg', 'rb') as photo:
            bot.send_photo(message.chat.id, photo=photo)
        os.remove('mrt.jpg')

        answer = 'Есть образование в мозге' if preds > 0.5 else 'Опухоль на предсотавленном снимке не обнаружена'

        bot.send_message(message.chat.id, text=f'Предсказания модели: {answer}')
        user_states[message.chat.id] = None
    except Exception as e:
        print('e')

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    #загрузка данных в таблицы по активности пользователя и предсказаниям
    text = f'Использование модели TWCLow, предсказание: {preds}'

    query = "INSERT INTO user_activity (user_id, activity, time) VALUES (?, ?, datetime('now'))"
    data = (message.chat.id, text)
    cursor.execute(query, data)

    query = "INSERT INTO predictions (user_id, predictions, answer, time) VALUES (?, ?, datetime('now'))"
    data = (message.chat.id, preds, answer)
    cursor.execute(query, data)

    conn.commit()
    conn.close()

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

    text = 'Обращение в тех.поддержку'
    query = "INSERT INTO user_activity (user_id, activity, time) VALUES (?, ?, datetime('now'))"
    data = (message.chat.id, text)

    cursor.execute(query, data)

    conn.commit()
    conn.close()

    work(message)

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

#функция по написаню отызва
@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == WAITING_REVIEW)
def review_user(message):
    print('review')
    text = message.text
    print(text)

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = "INSERT INTO review (user_id, user_review) VALUES (?, ?)"
    data = (message.chat.id, text)
    cursor.execute(query, data)

    conn.commit()
    conn.close()



#функции админа
#функция для отправки файлов из таблицы users
def info_user_bd(call):
    print('bd read work')

    # conn = sqlite3.connect('bot_base.db')
    # cursor = conn.cursor()
    # df = pd.read_sql_query("SELECT * FROM users", conn)
    # df.to_csv('/Bot/bd_users.csv', index=False)
    # df.to_excel('/Bot/bd_users.xlsx', index=False, engine='openpyxl')

    # conn.commit()
    # conn.close()

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

    cursor.execute(f"SELECT * FROM users WHERE user_id={text[0]}")
    result = cursor.fetchone()

    if result:
        markup = types.InlineKeyboardMarkup()
        button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
        markup.add(button)
        bot.send_message(message.chat.id, text='Такой пользоваетль уже есть', reply_markup=markup)

    else:
        cursor.execute("INSERT INTO users (user_id, user_name, login, password, role) VAlUES (?, ?, ?, ?, ?)",
                       (text[0], text[1], text[2], text[3], text[4]))

        markup = types.InlineKeyboardMarkup()
        button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
        markup.add(button)
        bot.send_message(message.chat.id, text='Пользователь добавлен', reply_markup=markup)


    conn.commit()
    conn.close()

    # print(text)


def user_edit(message):
    print('user edit')

    text = message.text
    text = text.split(':')

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM users WHERE user_id={text[0]}")
    result = cursor.fetchone()

    cursor.execute(f"UPDATE users SET login=? AND password=? AND role=? WHERE user_id='{text[0]}",
                   (text[0], text[1], text[2]))
    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(message.chat.id, text='Пользователь изменён', reply_markup=markup)

def bot_settings_admin(call):
    print('ueser settings bot')

    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('CSV', callback_data='bot_settings_admin_csv')
    button2 = types.InlineKeyboardButton('XLSX', callback_data='bot_settings_admin_xlsx')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

    markup.add(button1, button2, button)
    bot.send_message(call.message.chat.id, text='В каком формате предоставить данные по настройкам бота?', reply_markup=markup)

def bot_settings_admin_csv(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM bot_settings", conn)
    df.to_csv('bd_bot_settings.csv', index=False)

    # Открываем файл и отправляем его пользователю
    with open('bd_bot_settings.csv', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список настроек бота")
        os.remove('bd_bot_settings.csv')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)

def bot_settings_admin_xlsx(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM bot_settings", conn)
    df.to_excel('bd_bot_settings.xlsx', index=False, engine='openpyxl')

    # Открываем файл и отправляем его пользователю
    with open('bd_bot_settings.xlsx', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список настроек бота")
        os.remove('bd_bot_settings.xlsx')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)

#функция админа отправки файлово по отзывам пользователей
def review_admin(call):
    print('review')

    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('CSV', callback_data='admin_review_csv')
    button2 = types.InlineKeyboardButton('XLSX', callback_data='admin_review_xlsx')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

    markup.add(button1, button2, button)
    bot.send_message(call.message.chat.id, text='В каком формате предоставить данные?', reply_markup=markup)

#глубокуя функция отправки csv
def admin_review_csv(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM review", conn)
    df.to_csv('bd_review.csv', index=False)

    # Открываем файл и отправляем его пользователю
    with open('bd_review.csv', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список отзывов")
        os.remove('bd_review.csv')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)

def dmin_review_xlsx(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM review", conn)
    df.to_excel('bd_review.xlsx', index=False, engine='openpyxl')

    # Открываем файл и отправляем его пользователю
    with open('bd_review.xlsx', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список отзывов")
        os.remove('bd_review.xlsx')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)


def question(call):
    print('quastion')

    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('CSV', callback_data='admin_question_csv')
    button2 = types.InlineKeyboardButton('XLSX', callback_data='admin_question_xlsx')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

    markup.add(button1, button2, button)
    bot.send_message(call.message.chat.id, text='В каком формате предоставить данные?', reply_markup=markup)

#глубокуя функция отправки файла по частым вопросам пользователей в формате csv
def admin_question_csv(call):
    # print('admin_question_csv')
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM question", conn)
    df.to_csv('bd_question.csv', index=False)

    # Открываем файл и отправляем его пользователю
    with open('bd_question.csv', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список отзывов")
        os.remove('bd_question.csv')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)

#глубокуя функция отправки файла по частым вопросам пользователей в формате xlsx
def admin_question_xlsx(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM question", conn)
    df.to_excel('bd_question.xlsx', index=False, engine='openpyxl')

    # Открываем файл и отправляем его пользователю
    with open('bd_question.xlsx', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список отзывов")
        os.remove('bd_question.xlsx')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)

#отправка сообщения пользователю
def dialogs(call):
    print('dialog')

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = "SELECT user_id, user_name, role FROM users"
    cursor.execute(query)
    result = cursor.fetchall()

    user_list = []

    for row in result:
        print(row)
        user_list.append(row)
    # print(user_list[1])

    bot.send_message(call.message.chat.id, text='Текущие пользователи')
    for i in range(len(user_list)):
        bot.send_message(call.message.chat.id, text=f'{user_list[i][0]}:{user_list[i][1]} - {user_list[i][2]}')
    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton("Отправить сообщени", callback_data='send_admin_message')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button1, button)


    bot.send_message(call.message.chat.id, text='Записать сообщение?', reply_markup=markup)

@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == ADMIN_WRITE_USER)
def send_admin_message(message):
    print('send_admin_message')
    text = message.text
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = "INSERT INTO communication (user_id, user_name, user_role, user_message, time) VALUES (?,?,?,?, datetime('now'))"
    data = (str(message.chat.id), str(message.from_user.first_name), 'admin', str(text))
    cursor.execute(query, data)

    text = text.split(":")

    bot.send_message(text[0], text=f'{text[1]}')

    conn.commit()
    conn.close()

#функции разработчика
def bot_settings(call):
    print('bot settings')

    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('CSV', callback_data='bot_settings_developer_csv')
    button2 = types.InlineKeyboardButton('XLSX', callback_data='bot_settings_developer_xlsx')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

    markup.add(button1, button2, button)
    bot.send_message(call.message.chat.id, text='В каком формате предоставить данные?', reply_markup=markup)



def bot_settings_developer_csv(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM bot_settings", conn)
    df.to_csv('bd_bot_settings.csv', index=False)

    # Открываем файл и отправляем его пользователю
    with open('bd_bot_settings.csv', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список настроек бота")
        os.remove('bd_bot_settings.csv')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)

def bot_settings_developer_xlsx(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM bot_settings", conn)
    df.to_excel('bd_bot_settings.xlsx', index=False, engine='openpyxl')

    # Открываем файл и отправляем его пользователю
    with open('bd_bot_settings.xlsx', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список настроек бота")
        os.remove('bd_bot_settings.xlsx')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)

def bot_predict(call):
    print('predict')

    #взятие данных по предсказаниям
    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('CSV', callback_data='bot_predictions_developer_csv')
    button2 = types.InlineKeyboardButton('XLSX', callback_data='bot_predictions_developer_xlsx')
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')

    markup.add(button1, button2, button)
    bot.send_message(call.message.chat.id, text='В каком формате предоставить данные?', reply_markup=markup)

    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = "SELECT * FROM predictions"
    cursor.execute(query)
    result = cursor.fetchall()

    conn.commit()
    conn.close()

def bot_predictions_developer_csv(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    df.to_csv('bd_bot_predictions.csv', index=False)

    # Открываем файл и отправляем его пользователю
    with open('bd_bot_settings.csv', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список предсказаний бота")
        os.remove('bd_bot_settings.csv')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)

def bot_predictions_developer_xlsx(call):
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    df.to_excel('bd_bot_predictions.xlsx', index=False, engine='openpyxl')

    # Открываем файл и отправляем его пользователю
    with open('bd_bot_predictions.xlsx', 'rb') as f:
        bot.send_document(call.message.chat.id, f, caption="Вот список предсказаний бота")
        os.remove('bd_bot_predictions.xlsx')

    conn.commit()
    conn.close()

    markup = types.InlineKeyboardMarkup()
    button = types.InlineKeyboardButton("Выход в меню", callback_data='exit')
    markup.add(button)
    bot.send_message(call.message.chat.id, text='Данные предсоатвлены', reply_markup=markup)


def model_struct(call):
    #обращение к датасету
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    query = 'SELECT * FROM model_structure'
    cursor.execute(query)

    result = cursor.fetchall()

    bot.send_message(call.message.chat.id, text=f'Классы используемые в основной модели:')
    for row in result:
        bot.send_message(call.message.chat.id, text=f'Класс {result[1]}:\n'
                                                    f'{result[2]}')

    work(call.message)
    conn.commit()
    conn.close()



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
    #функция выведет на экран разработчику списки пользователей,
    #разработчик сможет ввести сообщение в формате "id пользователя:сообщение"
    conn = sqlite3.connect('bot_base.db')
    cursor = conn.cursor()

    text = message.text

    #данные заносятся в таблицу коммуникаций для анализа в случае проблем с ботом
    query = "INSERT INTO communication (user_id, user_name, user_role, user_message, time) VALUES (?,?,?,?, datetime('now'))"
    data = (str(message.chat.id), str(message.from_user.first_name), 'developer', str(text))
    cursor.execute(query, data)

    text = text.split(':')


    query = f'SELECT * FROM help_history WHERE user_id = {text[0]}'
    cursor.execute(query)
    result = cursor.fetchone()
    if result:
        cursor.execute(f'INSERT INTO help_history (assistant_id, assistant_answer) VAlUES (?, ?)', (message.chat.id, text[1]))
        bot.send_message(text[0], text=f'Здарвствуйте, ответ тех. поддержки, на ваш запрос: {text[1]}')


    conn.commit()
    conn.close()

bot.infinity_polling()
