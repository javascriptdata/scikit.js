/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import React from 'react'
import clsx from 'clsx'
import styles from './HomepageFeatures.module.css'

type FeatureItem = {
  title: string
  image: string
  description: JSX.Element
}

const FeatureList: FeatureItem[] = [
  {
    title: 'Easy to Use',
    image: '/img/MLBulbCircuit.svg',
    description: (
      <>
        Scikit.js was designed to be a Typescript port of the popular
        scikit-learn python library. If you ever needed an easy way to deploy
        to JS environments (browser / phone), this is the library for you.
      </>
    )
  },
  {
    title: 'Focus on What Matters',
    image: '/img/machine-learning5.png',
    description: (
      <>
        Scikit.js lets you focus on building and shipping models that use a
        familiar API. It relies on the blazingly fast Tensorflow.js as an
        underlying data structure to make computations fast.
      </>
    )
  },
  {
    title: 'Powered by Typescript',
    image: '/img/MLBulbGear.svg',
    description: (
      <>
        Use the languages and tools you are familiar with to ship production
        machine learning models faster, with an API that makes it easy to
        deploy and extend.
      </>
    )
  }
]

function Feature({ title, image, description }: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img className={styles.featureSvg} alt={title} src={image} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  )
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  )
}
