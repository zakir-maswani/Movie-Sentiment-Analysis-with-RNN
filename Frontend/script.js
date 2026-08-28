const reviewInput = document.getElementById("review-input");
const predictBtn = document.getElementById("predict-btn");
const charCount = document.getElementById("char-count");
const errorBox = document.getElementById("error");
const resultBox = document.getElementById("result");
const resultEmoji = document.getElementById("result-emoji");
const resultLabel = document.getElementById("result-label");
const barFill = document.getElementById("bar-fill");
const resultConfidence = document.getElementById("result-confidence");

const API_URL = "/api/predict";

reviewInput.addEventListener("input", () => {
  charCount.textContent = `${reviewInput.value.length} characters`;
});

predictBtn.addEventListener("click", async () => {
  const review = reviewInput.value.trim();

  hideError();
  resultBox.classList.add("hidden");

  if (!review) {
    showError("Please enter a review first.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    renderResult(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Is the backend running?");
  } finally {
    setLoading(false);
  }
});

function renderResult(data) {
  const isPositive = data.sentiment === "positive";
  resultEmoji.textContent = isPositive ? "😊" : "😞";
  resultLabel.textContent = isPositive ? "Positive" : "Negative";
  resultLabel.style.color = isPositive ? "#22c55e" : "#ef4444";

  const positivePct = data.positive_probability * 100;
  barFill.style.left = `${positivePct}%`;

  resultConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confident`;

  resultBox.classList.remove("hidden");
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  predictBtn.textContent = isLoading ? "Analyzing..." : "Analyze Sentiment";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}
