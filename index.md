<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>This site is for testing purpose</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
<script>
  renderMathInElement(document.body,{delimiters: {left: "$$", right: "$$", display: true}]});
  renderMathInElement(document.body,{delimiters: {left: "$", right: "$", display: false}]});
</script>
</head>
<body>





## Regression

The most simple model to start with is linear regression. In the regression exercises you will build a model, which predicts a floating point value (target value), based on only one or more other floating point values (features). You will get familiar with the concept of hypotheses, cost functions (here mean squared error), the gradient descent algorithm and the iterative update rule. 

### Univariate Linear Regression

Imagine you want to predict the house price (target value $y$), based sololy on one feature $x$, e.g. the area in sqm. You get some example data $D_{Train} = \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(m)}, y^{(m)})\}$, which consists of pairs of $(x^{(i)}, y^{(i)})$. Unfortunately, the data does not contain any example, which has exactly the sqm you want to know the price for. The most simple thing to do is to fit a straight line. Intuitively you have no problem to fit a line to the data. But how to tell an algorithm what a good line (model / hypothesis) is? First we need to know how to define a line in general: 

$$ h_{\theta}(x) = \theta_0 + \theta_1 \cdot x $$,

$\theta_0$ is the bias and $\theta_1$ is the slope. The next thing we need is a meassure that tells us how well a specific line fits to the data: a so called cost function $J(\theta)$. For regression we use the mean squared error: 

$$ J(\theta) = mse(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)} - y^{(i)})^2 $$,

with $m$ the number of our training examples. Squaring the distance between our prediciton $h_{\theta}(x^{(i)})$ and the true target value $y^{(i)}$ has two effects: We always get a positive value and bigger distances produce a much higher cost than smaller distances. Using the cost function we can now try different combinations of values for  $\theta_0$ and $\theta_1$ and draw an error-surface, which depicts the costs depending on  $\theta_0$ and $\theta_1$ and therefore how well each






</body>
</html>
