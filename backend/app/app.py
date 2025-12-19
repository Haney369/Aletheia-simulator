from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Aletheia backend is alive 🧠🔥"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("🚀 Starting Flask on port", port)
    app.run(host="0.0.0.0", port=port)
