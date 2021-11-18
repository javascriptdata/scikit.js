---
sidebar_position: 3
---

# Coming from python?

This library aims to be a drop-in replacement for scikit-learn but for JS environments. There are some
differences in deploy environment and underlying libraries that make for a slightly different experience.
Here are the 3 main differences.

**1. Class constructors take in objects. Every other function takes in positional arguments.**

While I would have liked to make every function identical to the python equivalent, it wasn't possible. In python,
one has named arguments, meaning that all of these are valid function calls.

#### python

```py
def myAdd(a=0, b=100):
  return a+b

print(myAdd()) # 100
print(myAdd(a=10)) # 110
print(myAdd(b=10)) # 10
print(myAdd(b=20, a=20)) # 40 (order doesn't matter)
print(myAdd(50,50)) # 100
```

Javascript doesn't have named parameters, so one must choose between positional arguments, or passing in a single object with all the parameters.

For many classes in scikit-learn, the [constructors take in a ton of arguments](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) with sane defaults, and the user usually only specifies which one they'd like to change. This rules out the positional approach.

After a class is created most function calls really only take in 1 or 2 arguments (think fit, predict, etc). In that case, I'd rather simply pass them positionally. So to recap.

#### python

```py
from sklearn.linear_model import LinearRegression

X, y = [[1],[2]], [10, 20]
lr = LinearRegression(fit_intercept = False)
lr.fit(X, y)
```

Turns into

#### javascript

```js
import { LinearRegression } from 'scikitjs'

let X = [[1], [2]]
let y = [10, 20]
let lr = new LinearRegression({ fitIntercept: false })
lr.fit(X, y)
```

You'll also notice in the code above, these are actual classes in JS, so you'll need to `new` them.

**2. underscore_case turns into camelCase**

Not a huge change, but every function call and variable name that is `underscore_case` in python will simply be `camelCase` in JS. In cases where there is an underscore but no word after, it is removed.

#### python

```py
from sklearn.linear_model import LinearRegression

X, y = [[1],[2]], [10, 20]
lr = LinearRegression(fit_intercept = False)
lr.fit(X, y)
print(lr.coef)
```

Turns into

#### javascript

```js
import { LinearRegression } from 'scikitjs'

let X = [[1], [2]]
let y = [10, 20]
let lr = new LinearRegression({ fitIntercept: false })
lr.fit(X, y)
console.log(lr.coef)
```

In the code sample above, we see that `fit_intercept` turns into `fitIntercept` (and it's an object). And `coef` turns into `coef`.

**3. The `fit` function for some estimators will be asynchronous, and so it will be called `fitAsync`**

In Javascript there are many cases where you can't tie up the main thread. In those cases it's best to use async functions.

Moreover some underlying libraries (tensorflow.js) only provide a `fit` function that is asynchronous. So if you wish to use those libraries, your `fit` function will also be async. But I didn't want to ship a library where users didn't easily know whether a function was asynchronous or not, so I opted to with the following convention:

- If your `fit` is synchronous, it is called `fit`.
- If it is async, then it is called `fitAsync`.

#### python

```py
from sklearn.linear_model import LogisticRegression

X, y = [[1],[-1]], [1, 0]
lr = LogisticRegression(fit_intercept = False)
lr.fit(X, y)
print(lr.coef)
```

Turns into

#### javascript

```js
import { LogisticRegression } from 'scikitjs'

let X = [[1], [-1]]
let y = [1, 0]
let lr = new LogisticRegression({ fitIntercept: false })
await lr.fitAsync(X, y)
console.log(lr.coef)
```

Why exactly is a `LinearRegression` synchronous, while a `LogisticRegression` is not? Well in the case of a `LinearRegression`, there exist [closed form](https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/) solutions that compute the proper coefficients. That is not the case in a `LogisticRegression`.

In the case of a `LogisticRegression` I opt for a SGD solution using the underlying `tensorflow.js` library for speed. The `fit` function that `tensorflow.js` gives me is async, and so therefore, mine is as well.
