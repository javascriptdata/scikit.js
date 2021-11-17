---
sidebar_position: 6
---

# Why?

### Why create a Typescript clone of scikit-learn? Shouldn't you just use the python scikit-learn?

I admit I'm a bit of a fanboy when it comes to the python scikit-learn. I think the API is fantastic. I think the models are performant, and the docs are great. A lot of the design of this library came from the idea that it was very easy to build a model in scikit-learn but harder to deploy it to web and mobile.

### Well can't you use ONNX?

You absolutely can. And in many cases where performance is absolutely critical it's best if you do. But I can't help feeling like there is something missing from the "convert your model to ONNX" train of thought. Let me give an example.

Let's say you are a real estate agency that owns a bunch of houses and you'd like to predict which ones will get sold by the end of the year. Here's an example historical dataset.

```csv
SquareFoot, Price, bathrooms, bedrooms, isSold
3000, 485k, 3, 3, 1
3320, 520k, 4, 2, 1
2800, 700k, 2, 2, 0
```

And so you diligently perform your duties as a data scientist.

```py
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./historicalData.csv')
y = df['isSold']
X = df.drop('isSold')

lr = LogisticRegression()
lr.fit(X, y)
```

But you want to deploy this model to the frontend, so that users who are setting the price of their homes know what they should sell at if they want to actually get their house sold. So let's use ONNX to build that model.

Let's also assume for this task that there is a form on the frontend that has 4 inputs (SquareFoot, Price, Bathrooms, Bedrooms) that is used to feed data to our model.

```py
from skl2onnx import to_onnx
...
lr = LogisticRegression()
lr.fit(X, y)
onx = to_onnx(lr, X[:1].astype(numpy.float32), target_opset=12)
```

Simple enough. Now we need to import the sklearn to onnx library, and convert to the ONNX format. I'm skimming over some things like "what is the target_opset?", and "why do we need to pass the input data to ONNX", but such is life. Our boss is happy, cause we can take this file, and then use the ONNX runtime inference to deploy to the browser.

A couple of days later, our boss comes back in and says the model isn't doing as well as it should. Could we spend some time and make it more accurate? Fine, no big. We sit down with the data and realize that we can construct another feature which is very predictive. It's called price_per_square_foot.

```py
from skl2onnx import to_onnx
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./historicalData.csv')
df['price_per_square_foot'] = df['Price'] / df['SquareFoot']
y = df['isSold']
X = df.drop('isSold')

lr = LogisticRegression()
lr.fit(X, y)
onx = to_onnx(lr, X[:1].astype(numpy.float32), target_opset=12)

```

But now when you deploy the model to the frontend, there is are now 5 inputs instead of 4. And so when the user adds in their information, we would need more JS on the frontend to make the model work (add an extra step that does the division of price and square_foot). So even if we use ONNX to deploy to the frontend, we still have to redo some of our machine learning in that domain because **all preprocessing is part of our model**.

So do you see my problem? If we venture outside of the preprocessors in scikit-learn we basically have to implement them on the JS side. Why not just write the model generation on the JS side? That way we can save ourselves the implementation of the preprocessing in JS (and the conversion with ONNX). All we need is a library that uses the same functions that we have in scikit-learn and we are good to go.

### But why Javascript?

When I started my career in machine learning 10 years ago, I lived and breathed python. I ate and slept python. I followed all of the work that was emerging then that python was moving to the browser (brython). I genuinely wished that I could just use python in the browser so that I could take these machine learning models, and put them into websites easily. As many of these projects came and went, I realized one fundamental truth: Javascript isn't going anywhere.

So I learned it. And built a company around it. And anytime I needed to do machine learning I would use python. But this added complexity to the organization **as a whole**. Anytime the frontend devs thought up a new feature to be used in the model, I'd have to implement it. And anytime the data scientists came up with a new model / architecture on the backend, big frontend changes would have to be used to make it.

But there is a simplification of the organization as a whole if you could just use one language. No more context switching between languages. No more disparities between the frontend and the model. A single dev could add a feature to the model, and update the UI to capitalize on it. It's full stack engineering but bringing machine learning into the fold.

People will routinely dismiss Domain Specific Languages (instead of templating, use React), but if your organization only uses python to do machine learning, then python is itself a Domain Specific Language. It's only used for machine learning and therefor needs experts that know it, while the rest of your organization uses a different language to do its business. In this case, it's best to simply change the way you write models so that you can write code in the language that the rest of your language uses. Less languages because then more people can look at the model.

I'm not alone in this thought. I just want the libraries in JS to match those in python so that it truly doesn't matter which one you use, hence the decision to make **scikit.js**.
