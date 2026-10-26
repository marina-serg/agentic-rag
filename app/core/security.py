import ipaddress
import re

from app.core.config import get_settings


class SecurityError(Exception):
    """Ошибка проверки входных или выходных данных."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


PATTERNS = {
    "email": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    ),
    "phone": re.compile(
        r"(?<!\w)(?:\+7|8)[\s(.-]?"
        r"(?:\d[\s).()-]?){10}(?!\d)"
    ),
    "snils": re.compile(
        r"(?<!\d)\d{3}-\d{3}-\d{3}\s\d{2}(?!\d)"
    ),
    "inn": re.compile(
        r"(?<!\d)(?:\d{10}|\d{12})(?!\d)"
    ),
    "passport": re.compile(
        r"(?<!\d)\d{2}\s?\d{2}\s?\d{6}(?!\d)"
    ),
    "card": re.compile(
        r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"
    ),
    "ip": re.compile(
        r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"
    ),
}


def _luhn_valid(value: str) -> bool:
    """Проверяет контрольную сумму номера карты."""

    digits = [int(char) for char in value if char.isdigit()]

    if not 13 <= len(digits) <= 19:
        return False

    total = 0

    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit

    return total % 10 == 0


def mask_pii(text: str) -> str:
    """Маскирует распознаваемые персональные данные."""

    text = PATTERNS["email"].sub("[EMAIL]", text)
    text = PATTERNS["snils"].sub("[SNILS]", text)
    text = PATTERNS["passport"].sub("[PASSPORT]", text)
    text = PATTERNS["inn"].sub("[INN]", text)

    def replace_card(match: re.Match) -> str:
        value = match.group(0)
        return "[CARD]" if _luhn_valid(value) else value

    text = PATTERNS["card"].sub(replace_card, text)

    def replace_ip(match: re.Match) -> str:
        value = match.group(0)
        try:
            ipaddress.ip_address(value)
            return "[IP]"
        except ValueError:
            return value

    text = PATTERNS["ip"].sub(replace_ip, text)
    text = PATTERNS["phone"].sub("[PHONE]", text)

    return text


def check_input(text: str) -> str:
    """Проверяет пользовательское сообщение."""

    if not isinstance(text, str) or not text.strip():
        raise SecurityError(
            "empty_input",
            "Сообщение не должно быть пустым",
        )

    settings = get_settings()
    text = text.strip()

    if len(text) > settings.max_input_length:
        raise SecurityError(
            "input_too_long",
            "Сообщение слишком длинное",
        )

    return text


def check_output(text: str) -> str:
    """Проверяет ответ модели перед отправкой."""

    if not isinstance(text, str) or not text.strip():
        raise SecurityError(
            "empty_output",
            "Модель вернула пустой ответ",
        )

    settings = get_settings()
    text = text.strip()

    if len(text) > settings.max_output_length:
        raise SecurityError(
            "output_too_long",
            "Ответ модели слишком длинный",
        )

    return text