import telebot
from telebot import types
import torchvision
import torch
from torchvision import transforms

from PIL import Image
from io import BytesIO

#токен для подключения к боту
token = '7433058915:AAHj5KtDTJ58OoGGUIayfWpDGOG3v0DnfgY'

#использование токена
bot = telebot.TeleBot(token)
user_states = {}


@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('Анализ снимка МРТ', 'Тех.поддержка')
    bot.send_message(
        message.chat.id,
        f"Привет, {message.from_user.first_name}! Выбери команду.",
        reply_markup=markup
    )

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    user_id = message.from_user.id
    text = message.text

    if text == 'Анализ снимка МРТ':
        user_states[user_id] = 'waiting_for_image'
        bot.send_message(message.chat.id, "Пожалуйста, отправьте изображение МРТ для анализа.")
        return
    elif text == 'Тех.поддержка':
        bot.send_message(message.chat.id, "Техподдержка в разработке.")
        return
    else:
        # Если ожидаем фото, но пришел текст
        if user_states.get(user_id) == 'waiting_for_image':
            bot.send_message(message.chat.id, "Пожалуйста, отправьте изображение для анализа.")
        else:
            bot.send_message(message.chat.id, "Я вас понял. Выберите команду.")

# Обработчик для фото
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    user_id = message.from_user.id
    if user_states.get(user_id) == 'waiting_for_image':
        try:
            # Получаем файл фото
            file_id = message.photo[-1].file_id
            file_info = bot.get_file(file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            # Создаем BytesIO объект
            image_stream = BytesIO(downloaded_file)

            # Анализируем изображение
            bot.send_message(message.chat.id, "Обрабатываю изображение, пожалуйста, подождите...")
            mrt_analis(image_stream, message)

            # После обработки сбрасываем состояние
            user_states[user_id] = None
        except Exception as e:
            bot.send_message(message.chat.id, f"Ошибка при обработке изображения: {e}")
    else:
        bot.send_message(message.chat.id, "Пожалуйста, сначала выберите команду для анализа.")

def mrt_analis(image_stream, message):
    model_path = '/Users/ilia/DeiT/DeiT_weights.pth'
    # Загружаем модель
    try:
        # Используйте вашу собственную реализацию DeiT
        model = DeiT(
            in_channels=3,
            patch_size=16,
            emb_size=384,  # важно совпадение с тем, что было при обучении
            img_size=224,
            depth=12,
            n_classes=2
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()


    except Exception as e:
        # bot.send_message(message.chat.id, f"Ошибка загрузки модели: {e}")
        print(e)
        return

    # Трансформы
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Предсказание
    def predict_image(image_stream):
        image = Image.open(image_stream).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
        return output

    predictions = predict_image(image_stream)
    # Тут можно обработать predictions и отправить результат
    bot.send_message(message.chat.id, f"Результат анализа: {str(predictions)}")
    print(predictions)
    print(1)

bot.infinity_polling()
