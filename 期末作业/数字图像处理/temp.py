
n = 5
s = 1
t = [0, 1, 3, 4, 2, 1]
w = [0, 3, 2, 3, 3, 4]

st = [0 for _ in range(n+2)]
sw = [0 for _ in range(n+2)]
z = [999 for _ in range(n+2)]
z[n+1] = 0
for i in range(n, 0, -1):
    st[i] = st[i+1] + t[i]
    sw[i] = sw[i+1] + w[i]

for i in range(n, 0, -1):
    for j in range(i+1, n+2):
        z[i] = min(z[i], z[j]+sw[i]*(st[i]+s-st[j]))
print(z[1])
