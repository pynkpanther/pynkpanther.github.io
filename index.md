<!doctype html>
<html>
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/contrib/auto-render.min.js"></script>
</head>
<body>















    <div>The formula $a^2+b^2=c^2$ will be rendered inline, but $$a^2+b^2=c^2$$ will be rendered as a block element.</div>
    <br>
    <div>The formula \(a^2+b^2=c^2\) will be rendered inline, but \[a^2+b^2=c^2\] will be rendered as a block element.</div>















    <script>
      renderMathInElement(
          document.body,
          {
              delimiters: [
                  {left: "$$", right: "$$", display: true},
                  {left: "\\[", right: "\\]", display: true},
                  {left: "$", right: "$", display: false},
                  {left: "\\(", right: "\\)", display: false}
              ]
          }
      );
    </script>
</body>
</html>
