---
sidebar_position: 3
---

# Coming from python?

This library aims to be a drop-in replacement for scikit-learn but for JS environments. There are some
differences in deploy environment and underlying libraries that make for a slightly different experience.
Here are the 3 main differences.

### 1. Class constructors take in objects. Every other function takes in positional arguments.

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
await lr.fit(X, y)
```

You'll also notice in the code above, these are actual classes in JS, so you'll need to `new` them.

### 2. underscore_case turns into camelCase

Not a huge change, but every function call and variable name that is `underscore_case` in python will simply be `camelCase` in JS. In cases where there is an underscore but no word after, it is removed.

#### python

```py
from sklearn.linear_model import LinearRegression

X, y = [[1],[2]], [10, 20]
lr = LinearRegression(fit_intercept = False)
lr.fit(X, y)
print(lr.coef_)
```

Turns into

#### javascript

```js
import { LinearRegression } from 'scikitjs'

let X = [[1], [2]]
let y = [10, 20]
let lr = new LinearRegression({ fitIntercept: false })
await lr.fit(X, y)
console.log(lr.coef)
```

In the code sample above, we see that `fit_intercept` turns into `fitIntercept` (and it's an object). And `coef_` turns into `coef`.

### 3. Always await calls to .fit or .fitPredict

It's common practice in Javascript to not tie up the main thread. Many libraries, including tensorflow.js only give an async "fit" function.

So if we build on top of them our fit functions will be asynchronous. But what happens if we make our own estimator that has a synchronous fit function? Should we burden the user with finding out if their fit function is async or not, and then "awaiting" the proper one? I think not.

I think we should simply await all calls to fit. If you await a synchronous function, it resolves immediately and you are on your merry way. So I literally await all calls to .fit and you should too.

#### python

```py
from sklearn.linear_model import LogisticRegression

X, y = [[1],[-1]], [1, 0]
lr = LogisticRegression(fit_intercept = False)
lr.fit(X, y)
print(lr.coef_)
```

Turns into

#### javascript

```js
import { LogisticRegression } from 'scikitjs'

let X = [[1], [-1]]
let y = [1, 0]
let lr = new LogisticRegression({ fitIntercept: false })
await lr.fit(X, y)
console.log(lr.coef)
```
