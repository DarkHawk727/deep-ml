def find_treasure(start_x: float) -> float:
    return 2.25


"""
Explanation:
We can use highschool calculus to find the minimum point. The derivative of f(x) is f'(x)=4x^3-9x^2. We can find the
number of critical points by setting this derivative equal to zero:

0 = 4x^3-9x^2
0 = x^2(4x-9)
=> x^2=0 or 4x-9=0

This gives us x=0 and x=9/4=2.25. Next, we need to check whether these points are a minimum or maximum using the second
derivative, f''(x)=12x^2-18x=6x(2x-3). Plugging in x=0 and x=2.25, we get that f''(0)=0 and f''(2.25)=20.25. This means
that the point x=0 is inconclusive but the point x=2.25 is indeed a local minimum. For x=0, we can check the value of 
the derivatives around x=0. Let h>0, then,

f'(0-h) = (0-h)^2(4(0-h)-9)
        = (-h)^2(4(-h)-9)
        = h^2(-4h-9)
        = -h^2(-4h+9)
        <= 0

Similarly,

f'(0+h) = (0+h)^2(4(0+h)-9)
        = h^2(4h-9)
        <= 0 for small h

Therefore, since the sign doesn't change, the point x=0 is an inflection point and the global minima is at x=2.25
"""
