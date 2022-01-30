---
sidebar_position: 4
---

# Let's build a model predicting company churn

Here we are going to walk through an example where we use scikit.js to predict churn at a major telecomunications company. We will be using the dataset here (Kaggle link).

It's hosted remotely here (though we should probably host it as well).

## Examining the data

First let's create a new directory called `telco` that we are going to perform JS magic in. Let's also initialize npm, so that we can download our packages

> > > mkdir telco
> > > cd telco
> > > npm init

For this analysis, we are going to use `danfojs-node`, and `scikitjs`. So let's install them.

> > > npm install danfojs-node scikitjs

```javascript
import * as dfd from 'danfojs-node'
import {
  LogisticRegression,
  OneHotEncoder,
  makeColumnTransformer,
  StandardScaler,
  makePipeline
} from 'scikitjs/node'

/**
 * First let's read in the data and take a look at it.
 */
let df = await dfd.readCSV(
  'https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

/**
 * The first thing I like to do is figure out the types of the input
 * We can scour the df.ctypes.print() to see what the types are, and also use
 * df.
 * It looks like there are some useless categories.
 * customerID has only unique values (none are repeated) and so therefor it
 * can be safely ignored.
 *
 * Let's check for any missing values. It looks like there are none.
 *
 * By using the unique function we can easily see which columns are categorical
 * or not. Or my inspecting the danfo column types. df.ctypes.print()
 */

let categorical = [
  'gender',
  'Partner',
  'Dependents',
  'PhoneService',
  'Contract',
  'PaperlessBilling',
  'PaymentMethod',
  'MultipleLines',
  'InternetService',
  'OnlineSecurity',
  'TechSupport',
  'StreamingTV',
  'StreamingMovies',
  'OnlineBackup',
  'DeviceProtection'
]

let numerical = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

let y = df['Churn']

let colTransform = makeColumnTransformer([
  [new OneHotEncoder(), categorical],
  [new StandardScaler(), numerical]
])

let lr = makePipeline(colTransform, new LogisticRegression())
await lr.fit(df, y)
console.log(lr.coef)
console.log(lr.score())
```
