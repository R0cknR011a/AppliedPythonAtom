import sys
import json
import csv
# Ваши импорты
from file_extension import extension
from data_print import printer
if __name__ == '__main__':
    filename = sys.argv[1]

    # Ваш код
    encode = filename.split('-')[-1].split('.')[0]
    if extension(filename, encode) == 'json':
        with open(filename, encoding=encode) as f:
            data = json.load(f)
            m = [list(data[0].keys())]
            for i in data:
                m.append(list(i.values()))
            printer(m)
    elif extension(filename, encode) == 'tsv':
        with open(filename, encoding=encode) as f:
            data = csv.reader(f, delimiter='\t')
            printer(list(data))
    elif extension(filename, encode) is None:
        print('File extension not valid')
