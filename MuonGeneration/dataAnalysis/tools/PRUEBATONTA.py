list = [1, 2, 3, 4, 5]
# print(list)

a_0 = 1
d = 2
r = 2
for n in list:
    # a_n = a_0 + (n-1)*d
    a_n = a_0 * r**(n-1)
    print(a_n) 