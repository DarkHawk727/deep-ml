def dice_statistics(n: int) -> tuple[float, float]:
    expected_value = (n + 1) / 2
    variance = (n**2 - 1) / 12
    return expected_value, variance


E, V = dice_statistics(6)
print(round(E, 4), round(V, 4))  # 3.5 2.9167

"""
Let X be the random variable that represents the face of the die. Since the die is fair, the probability of any face is 
1/n. Then,

E[X] = Σ x * p(X=x)
     = Σ x * (1/n)
     = (1/n) Σ x
     = (1/n) * (n(n+1)) / 2
     = (n+1) / 2

Similarly, for the variance, we have,

Var(X) = E[X^2] - E[X]^2
       = [Σ x^2 * p(X=x)] - ((n+1) / 2)^2
       = [Σ x^2 * (1/n)] - (n+1)^2 / 4
       = [(1/n) Σ x^2] - (n+1)^2 / 4
       = (1/n) * (n(n+1)(2n+1)) / 6 - (n+1)^2 / 4
       = (n+1)(2n+1) / 6 - (n+1)^2 / 4
       = 2(n+1)(2n+1) / 12 - 3(n+1)^2 / 12
       = (2(n+1)(2n+1)-3(n + 1)^2) / 12
       = (n+1)(2(2n+1)-3(n+1)) / 12
       = (n+1)(4n+2-3n-3) / 12
       = (n+1)(n-1) / 12
       = (n^2 - 1) / 12
"""
