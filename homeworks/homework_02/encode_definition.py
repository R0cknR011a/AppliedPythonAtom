def is_utf8(filename):
    try:
        with open(filename, encoding='utf8') as f:
            f.read(1)
    except UnicodeError:
        return False
    return True


def is_cp1251(filename):
    try:
        with open(filename, encoding='cp1251') as f:
            f.read(1)
    except UnicodeError:
        return False
    return True


def is_utf16(filename):
    try:
        with open(filename, encoding='utf16') as f:
            f.read(1)
    except UnicodeError:
        return False
    return True


def define_encode(filename):
    if is_utf8(filename):
        return 'utf8'
    elif is_utf16(filename):
        return 'utf16'
    elif is_cp1251(filename):
        return 'cp1251'
    else:
        return None
