
document.addEventListener("DOMContentLoaded", () => {
    let messageForm = document.getElementById("messageForm");
    let mainChat = document.querySelector("main");
    let mainText = document.querySelector("#main");

    messageForm.addEventListener("submit", (event) => {
        event.preventDefault(); // Prevent form submission
        sendMessage(); // Handle sending message
    });

    function sendMessage() {
        let msgInput = document.querySelector("#msg");
        let inputText = msgInput.value.trim();

        if (inputText !== '') {
            let userMessage = createUserMessage(inputText);
            mainChat.appendChild(userMessage);
            mainText.style.display = "none"; // Hide mainText after first message

            fetch('/get_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: inputText }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.response) {
                    let botMessage = createBotMessage(data.response);
                    mainChat.appendChild(botMessage);
                } else {
                    let errorMessage = createBotMessage("Error: " + data.error);
                    mainChat.appendChild(errorMessage);
                }
                mainChat.scrollTop = mainChat.scrollHeight; // Scroll to bottom
            })
            .catch((error) => {
                console.error('Error:', error);
                let errorMessage = createBotMessage("Error: " + error);
                mainChat.appendChild(errorMessage);
            });

            msgInput.value = ''; // Clear input field after sending
        } else {
            alert("Please enter a message!");
        }
    }

    function createUserMessage(text) {
        let messageDiv = document.createElement("div");
        messageDiv.classList.add("user-message");
        messageDiv.innerText = text;
        return messageDiv;
    }

    function createBotMessage(text) {
        let messageDiv = document.createElement("div");
        messageDiv.classList.add("bot-message");
        messageDiv.innerText = text;
        return messageDiv;
    }
});