e = 0.995
decay = 0.994
for i in range(300):
    e = e*decay
    print(i, e)