import { Translations } from '../types'

export const ruTranslations: Translations = {
  // Header
  'app.title': 'Санскрит OCR и Конвертер',
  'app.subtitle': 'OCR, Обработка Текста и Конвертация Форматов',
  'app.description': 'Обработка изображений или текста • Конвертация между русской и IAST диакритикой • Поиск в базе данных',
  
  // Navigation
  'nav.ocr': 'OCR Распознавание',
  'nav.database': 'Управление БД',
  'nav.admin': 'Админ',
  
  // Input Types
  'input.type.title': 'Тип Ввода',
  'input.type.image': 'Изображение',
  'input.type.text': 'Текст',
  
  // Input Format
  'input.format.title': 'Формат Текста в Изображении',
  'input.format.english': 'Английская Диакритика',
  'input.format.russian': 'Русская Диакритика',
  'input.format.description': 'Выберите тип диакритики в вашем изображении: английская (ā ī ū ṛ ś ṣ) или русская (а̄ ӣ ӯ р̣ ш́)',
  
  // Output Format
  'output.format.title': 'Формат Вывода',
  'output.format.russian': 'Русская Диакритика',
  'output.format.iast': 'IAST (Английский)',
  
  // Image Upload
  'upload.drag.title': 'Перетащите ваше санскритское изображение сюда',
  'upload.drag.subtitle': 'или нажмите для выбора файла (макс. 10МБ)',
  'upload.select': 'Выбрать Изображение',
  'upload.processing': 'Обработка...',
  'upload.process': 'Обработать Изображение',
  
  // Text Input
  'text.input.label': 'Введите санскритский текст ({format} или любой формат)',
  'text.input.placeholder': 'Введите санскритский текст в любом формате (IAST, русская диакритика и т.д.)...',
  'text.characters': 'Символов: {count}',
  'text.process': 'Обработать Текст',
  
  // Results
  'results.corrected': 'Исправленный Текст (из Базы Данных)',
  'results.processed': 'Обработанный Текст ({format})',
  'results.copy': 'Копировать',
  'results.clear': 'Очистить',
  
  // Source Info
  'source.found': '📚 Источник Найден в Базе Данных',
  'source.book': 'Книга:',
  'source.chapter': 'Глава:',
  'source.verse': 'Стих:',
  'source.confidence': 'Точность:',
  'source.match': 'Тип Совпадения:',
  
  // Database Stats
  'stats.texts': 'Тексты',
  'stats.books': 'Книги',
  'stats.unique_words': 'Уникальные Слова',
  'stats.total_words': 'Всего Слов',
  
  // Admin Panel
  'admin.access.granted': '✅ Административный Доступ Предоставлен',
  'admin.access.description': 'Теперь вы можете загружать книги в базу данных.',
  'admin.logout': 'Выйти',
  'admin.access.required': '🔐 Требуется Административный Доступ',
  'admin.access.restriction': 'Загрузка книг доступна только администраторам. Войдите для доступа к функциям управления базой данных.',
  'admin.login': 'Вход Админа',
  'admin.cancel': 'Отмена',
  'admin.password.placeholder': 'Введите пароль администратора',
  'admin.logging': 'Вход...',
  'admin.login.button': 'Войти',
  
  // Book Upload
  'book.upload.title': 'Перетащите ваши санскритские книги сюда',
  'book.upload.subtitle': 'Поддерживает форматы PDF, DOCX, TXT, DOC, RTF, ODT (макс. 50МБ)',
  'book.select': 'Выбрать Книгу',
  'book.upload.processing': 'Обработка...',
  'book.upload.button': 'Загрузить Книгу',
  'book.upload.success': 'Успешно добавлено {count} текстов из {filename}',
  
  // Footer
  'footer.description': 'OCR Изображений • Обработка Текста • Конвертация Форматов (IAST ⇄ Русский) • Поиск в БД • Множественные Методы Ввода',
  'footer.contact': 'По всем вопросам обращайтесь через Telegram или email:',
  
  // Error Messages
  'error.image.format': 'Пожалуйста, выберите файл изображения',
  'error.file.size': 'Размер файла должен быть менее 10МБ',
  'error.file.size.book': 'Размер файла должен быть менее 50МБ',
  'error.document.format': 'Пожалуйста, выберите поддерживаемый файл документа (PDF, DOCX, TXT, DOC, RTF, ODT)',
  'error.process.image': 'Не удалось обработать изображение',
  'error.process.text': 'Не удалось обработать текст',
  'error.process.book': 'Не удалось обработать книгу',
  'error.copy.clipboard': 'Не удалось скопировать текст в буфер обмена'
} 