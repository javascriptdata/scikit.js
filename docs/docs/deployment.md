---
sidebar_position: 3
---

# Deployment (Isomorphic ML)

Wouldn't it be nice if you could train a model, and then choose whether you wanted to deploy it to the backend or the frontend? A lot of my last 10 years in ML has shown that we really need to ease the burden that it takes to go from idea to production.

I've been working in machine learning my whole career, and one of the things I've noticed is just how hard it is to deploy machine learning models. Over 87% of machine learning models never make it to [production](https://venturebeat.com/2019/07/19/why-do-87-of-data-science-projects-never-make-it-into-production/).

There's even a whole role at an organization that just moves models from research to production. Usually that person is called an ML engineer.

In my dream world, I'd like to

1. Make a simple create-react-app
2. Train a model in the backend for speed
3. Place the model in the public directory
4. Then consume that model in the frontend

So let's make some dreams come true.
