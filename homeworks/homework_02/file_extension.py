import json
import csv


def check_tsv(filename, encode):
    with open(filename, encoding=encode) as f:
        data = csv.reader(f, delimiter='\t')
        m = []
        for i in data:
            m.append(len(i))
        if m.count(m[0]) == len(m):
            return True
        else:
            return False


def check_json(filename, encode):
    try:
        with open(filename, encoding=encode) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False
    return True


def extension(filename, encode):
    if check_json(filename, encode):
        return 'json'
    elif check_tsv(filename, encode):
        return 'tsv'
    else:
        raise SystemExit('Invalid extension')
