from flask import Flask, render_template, request, jsonify
from chatbot import get_response, process_query

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/get_response', methods=['POST'])
def handle_response():
    try:
        user_message = request.json.get("message")
        print(f"Received message: {user_message}")  # Debugging line
        if user_message:
            response, entities = process_query(user_message)
            print(f"Response: {response}, Entities: {entities}")  # Debugging line
            return jsonify({"response": response, "entities": entities}), 200
        else:
            return jsonify({"error": "Empty message received"}), 400
    except Exception as e:
        print(f"Error: {e}")  # Debugging line
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
