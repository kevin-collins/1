n = 7
k = 3
t = [2, 14, 4, 16, 6, 5, 3]
tk = [0 for _ in range(k)]


def backtrack(i, best):
    if i >= n:
        best = min(best, max(tk))
        return best
    for j in range(k):
        tk[j] += t[i]
        if tk[j] < best:
            best = backtrack(i+1, best)
        tk[j] -= t[i]
    return best

print(backtrack(0, 9999999))

