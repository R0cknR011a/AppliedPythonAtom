def printer(data):
    m = []
    for i in range(len(data[0])):
        max = 0
        for j in data:
            if type(j[i]) is str and len(j[i]) >= max:
                max = len(j[i])
        m.append(max)
    print('-' * (sum(m) + (len(data[0]) - 1) * 5 + 6))
    for i in data:
        k = 0
        for j in i:
            if data.index(i) == 0:
                print('|  ' + ' ' * ((m[k] - len(j))//2) + j +
                      ' ' * ((m[k] - len(j))//2), end='  ')
            else:
                if j.isdigit():
                    print('|  ' + ' ' * (m[k] - len(j)) + j, end='  ')
                else:
                    print('|  ' + j + ' ' * (m[k] - len(j)), end='  ')
            k += 1
        print('|')
    print('-' * (sum(m) + (len(data[0]) - 1) * 5 + 6))
