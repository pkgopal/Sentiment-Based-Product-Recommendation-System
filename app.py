# ==============================
# app.py
# ==============================

from flask import Flask, request, render_template_string
from model import recommend_products

app = Flask(__name__)

# ==============================
# HTML TEMPLATE (Embedded)
# ==============================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Sentiment-Based Product Recommendation</title>
<style>
body {
font-family: Arial, sans-serif;
margin: 40px;
background-color: #f4f6f9;
}
h2 {
color: #2c3e50;
}
form {
margin-bottom: 20px;
}
input[type=text] {
padding: 8px;
width: 250px;
}
input[type=submit] {
padding: 8px 15px;
background-color: #3498db;
color: white;
border: none;
cursor: pointer;
}
input[type=submit]:hover {
background-color: #2980b9;
}
ul {
background: white;
padding: 15px;
border-radius: 5px;
width: 400px;
}
li {
margin-bottom: 8px;
}
</style>
</head>
<body>

<h2>Sentiment-Based Product Recommendation System</h2>

<form method="POST">
<label>Enter Username:</label><br><br>
<input type="text" name="username" required>
<input type="submit" value="Recommend">
</form>

{% if recommendations %}
<h3>Top 5 Recommended Products:</h3>
<ul>
{% for product in recommendations %}
<li>{{ product }}</li>
{% endfor %}
</ul>
{% endif %}

</body>
</html>
"""

# ==============================
# ROUTE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None

    if request.method == "POST":
        username = request.form.get("username").strip()
        recommendations = recommend_products(username)

    return render_template_string(HTML_TEMPLATE, recommendations=recommendations)


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":
    app.run(debug=False)