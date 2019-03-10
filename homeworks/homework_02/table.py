import sys
import json
import csv
# Ваши импорты
from file_extension import extension
from data_print import printer
from encode_definition import define_encode
if __name__ == '__main__':
    filename = sys.argv[1]

    # Ваш код
    try:
        f = open(filename)
        f.close()
    except FileNotFoundError:
        raise SystemExit('Invalid file')
    encode = define_encode(filename)
    if extension(filename, encode) == 'json':
        with open(filename, encoding=encode) as f:
            data = json.load(f)
            print(data)
            m = [list(data[0].keys())]
            for i in data:
                m.append(list(i.values()))
            printer(m)
    elif extension(filename, encode) == 'tsv':
        with open(filename, encoding=encode) as f:
            data = csv.reader(f, delimiter='\t')
            printer(list(data))
