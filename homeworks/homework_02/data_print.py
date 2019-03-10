def printer(data):
    m = []
    for i in range(len(data[0])):
        max = 0
        for j in data:
            if type(j[i]) is str and len(str(j[i])) >= max:
                max = len(str(j[i]))
        m.append(max)
    print('-' * (sum(m) + (len(data[0]) - 1) * 5 + 6))
    for i in data:
        k = 0
        for j in i:
            if data.index(i) == 0:
                print('|  ' + ' ' * ((m[k] - len(str(j)))//2) + str(j) +
                      ' ' * ((m[k] - len(str(j)))//2), end='  ')
            else:
                if str(j).isdigit():
                    print('|  ' + ' ' * (m[k] - len(str(j))) + str(j), end='  ')
                if type(j) is str:
                    print('|  ' + str(j) + ' ' * (m[k] - len(str(j))), end='  ')
            k += 1
        print('|')
    print('-' * (sum(m) + (len(data[0]) - 1) * 5 + 6))
