document.addEventListener("DOMContentLoaded", () => {
  const uploadBtn = document.getElementById("uploadBtn");
  const imageUpload = document.getElementById("imageUpload");
  const originalImage = document.getElementById("originalImage");
  const annotatedImage = document.getElementById("annotatedImage");
  const prediction = document.getElementById("prediction");
  const confidence = document.getElementById("confidence");
  const resultSection = document.getElementById("resultSection");
  const saveCommentBtn = document.getElementById("saveComment");
  const commentBox = document.getElementById("comment");

  uploadBtn.addEventListener("click", async () => {
    const file = imageUpload.files[0];
    if (!file) {
      alert("Please select an image");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("project_id", document.getElementById("projectID").value);
    formData.append("project_description", document.getElementById("projectDesc").value);

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      // Show results
      originalImage.src = URL.createObjectURL(file);
      annotatedImage.src = `http://localhost:8000${result.annotated_path}`;
      prediction.textContent = result.prediction === "corrosion" ? "CORROSION DETECTED" : "NO CORROSION";
      prediction.style.color = result.prediction === "corrosion" ? "red" : "green";
      confidence.textContent = (result.confidence * 100).toFixed(1) + "%";

      resultSection.style.display = "block";
    } catch (err) {
      alert("Error: Could not process image");
      console.error(err);
    }
  });

  saveCommentBtn.addEventListener("click", () => {
    const comment = commentBox.value.trim();
    if (comment) {
      alert("Notes saved successfully!");
      console.log("Comment saved:", comment);
      // You can send this to your backend via API later
    } else {
      alert("Please add a comment before saving.");
    }
  });
});