document.getElementById("text-form").addEventListener("submit", function(event) {
    event.preventDefault();
    const textInput = document.getElementById("text-input").value;
    
    fetch("/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ description: textInput }),
    })
    .then(response => response.json())
    .then(data => {
        const imgElement = document.getElementById("generated-image");
        imgElement.src = data.image_url; // Update the image URL with backend result
        imgElement.style.display = "block";
    })
    .catch(error => console.error("Error generating image:", error));
});
