
 // ==========================================
 // iGROW DASHBOARD
 // ==========================================

 // LOGIN CHECK
 if (sessionStorage.getItem("loggedIn") !== "true") {
     window.location.replace("login.html");
 }

 // LOGOUT
 const logoutButton = document.getElementById("logoutButton");

 if (logoutButton) {
     logoutButton.addEventListener("click", function () {
         sessionStorage.removeItem("loggedIn");
         window.location.replace("login.html");
     });
 }

 // PROFIT & LOSS CHART
 const profitCanvas = document.getElementById("profitChart");

 if (profitCanvas) {
     const profitCtx = profitCanvas.getContext("2d");

     new Chart(profitCtx, {
         type: "line",
         data: {
             labels: ["June", "July", "August", "September"],
             datasets: [
                 {
                     label: "Revenue",
                     data: [40000, 55000, 70000, 85000],
                     tension: 0.3
                 },
                 {
                     label: "Expenses",
                     data: [18000, 21000, 25000, 32000],
                     tension: 0.3
                 },
                 {
                     label: "Profit",
                     data: [22000, 34000, 45000, 53000],
                     tension: 0.3
                 }
             ]
         },
         options: {
             responsive: true,
             maintainAspectRatio: false,
             plugins: {
                 legend: { position: "bottom" }
             }
         }
     });
 }

 // ==========================================
 // CORIANDER GROWTH CHART
 // ==========================================

 const growthCanvas = document.getElementById("growthChart");
 const cropSelect = document.getElementById("cropSelect");

 const growthData = {
     coriander: [8, 15, 24, 35, 47],
     lettuce: [5, 12, 20, 28, 34],
     spinach: [6, 14, 23, 31, 39]
 };

 let growthChart;

 if (growthCanvas) {
     const growthCtx = growthCanvas.getContext("2d");

     const initialCrop =
         cropSelect && growthData[cropSelect.value]
             ? cropSelect.value
             : "coriander";

     growthChart = new Chart(growthCtx, {
         type: "line",
         data: {
             labels: [
                 "Week 1",
                 "Week 2",
                 "Week 3",
                 "Week 4",
                 "Week 5"
             ],
             datasets: [
                 {
                     label:
                         initialCrop.charAt(0).toUpperCase() +
                         initialCrop.slice(1) +
                         " Growth (cm)",
                     data: growthData[initialCrop],
                     tension: 0.3
                 }
             ]
         },
         options: {
             responsive: true,
             maintainAspectRatio: false,
             plugins: {
                 legend: { position: "bottom" }
             }
         }
     });
 }

 // CROP SELECTION
 if (cropSelect && growthChart) {
     cropSelect.addEventListener("change", function () {
         const selectedCrop = this.value;

         if (!growthData[selectedCrop]) {
             return;
         }

         growthChart.data.datasets[0].data =
             growthData[selectedCrop];

         growthChart.data.datasets[0].label =
             selectedCrop.charAt(0).toUpperCase() +
             selectedCrop.slice(1) +
             " Growth (cm)";

         growthChart.update();
     });
 }

 // ==========================================
 // YIELD CHART
 // ==========================================

 const yieldCanvas = document.getElementById("yieldChart");

 if (yieldCanvas) {
     const yieldCtx = yieldCanvas.getContext("2d");

     new Chart(yieldCtx, {
         type: "bar",
         data: {
             labels: [
                 "Coriander",
                 "Lettuce",
                 "Spinach",
                 "Cucumber"
             ],
             datasets: [
                 {
                     label: "Yield (kg)",
                     data: [450, 300, 250, 380]
                 }
             ]
         },
         options: {
             responsive: true,
             maintainAspectRatio: false,
             plugins: {
                 legend: { display: false }
             }
         }
     });
 }

 // ==========================================
 // MARKET PRICING CHART
 // ==========================================

 const marketCanvas = document.getElementById("marketChart");

 if (marketCanvas) {
     const marketCtx = marketCanvas.getContext("2d");

     new Chart(marketCtx, {
         type: "line",
         data: {
             labels: ["June", "July", "August", "September"],
             datasets: [
                 {
                     label: "Price / kg",
                     data: [38, 42, 47, 44],
                     tension: 0.3
                 }
             ]
         },
         options: {
             responsive: true,
             maintainAspectRatio: false
         }
     });
 }

 
/* ==========================================
   AI CROP RECOMMENDATION
========================================== */

const cropForm = document.getElementById("cropForm");

if (cropForm) {
    cropForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const button = document.getElementById("recommendButton");
        const message = document.getElementById("cropMessage");
        const result = document.getElementById("cropResult");
        const cropName = document.getElementById("recommendedCrop");

        button.disabled = true;
        message.textContent = "Getting crop recommendation...";
        result.hidden = true;

        const formData = new FormData(cropForm);

        const payload = {
            temperature: Number(formData.get("temperature")),
            humidity: Number(formData.get("humidity")),
            ph: Number(formData.get("ph")),
            ec: Number(formData.get("ec")),
            light_hours: Number(formData.get("light_hours")),
            water: formData.get("water"),
            season: formData.get("season")
        };

        try {
            const response = await fetch(
                "http://127.0.0.1:5000/api/predict",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || `Server error: ${response.status}`
                );
            }

            if (!data.recommended_crop) {
                throw new Error(
                    "The server did not return a recommended_crop value."
                );
            }

            cropName.textContent = data.recommended_crop;
            result.hidden = false;
            message.textContent =
                "Crop recommendation generated successfully.";

        } catch (error) {
            console.error("Crop recommendation error:", error);

            message.textContent =
                error.name === "TypeError"
                    ? "Cannot connect to the prediction server. Start your Python backend and check the API address."
                    : "Recommendation failed: " + error.message;

        } finally {
            button.disabled = false;
        }
    });
}

 // ==========================================
 // CORIANDER PRICE PREDICTION API
 // ==========================================

 const priceForm = document.getElementById("priceForm");

 if (priceForm) {
     const priceButton =
         document.getElementById("pricePredictButton");
     const priceMessage =
         document.getElementById("priceMessage");
     const priceResult =
         document.getElementById("priceResult");
     const predictedPrice =
         document.getElementById("predictedPrice");
     const priceResultDate =
         document.getElementById("priceResultDate");
     const priceDate =
         document.getElementById("priceDate");

     // Default to today's local date.
     if (priceDate) {
         const today = new Date();
         const year = today.getFullYear();
         const month = String(today.getMonth() + 1).padStart(2, "0");
         const day = String(today.getDate()).padStart(2, "0");

         priceDate.value = `${year}-${month}-${day}`;
     }

     priceForm.addEventListener("submit", async (event) => {
         event.preventDefault();

         priceButton.disabled = true;
         priceMessage.textContent =
             "Predicting coriander price...";
         priceResult.hidden = true;

         const formData = new FormData(priceForm);

         const payload = {
             price_lag_1: Number(formData.get("price_lag_1")),
             price_lag_2: Number(formData.get("price_lag_2")),
             price_lag_3: Number(formData.get("price_lag_3")),
             price_lag_4: Number(formData.get("price_lag_4")),
             price_lag_5: Number(formData.get("price_lag_5")),
             price_lag_6: Number(formData.get("price_lag_6")),
             price_lag_7: Number(formData.get("price_lag_7")),
             temperature: Number(formData.get("temperature")),
             humidity: Number(formData.get("humidity")),
             rainfall: Number(formData.get("rainfall")),
             date: formData.get("date")
         };

         try {
             
const response = await fetch("http://127.0.0.1:5000/api/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
});

const data = await response.json();

if (!response.ok) {
    throw new Error(data.error || "Crop prediction failed");
}

console.log("Crop recommendation:", data.recommended_crop);

             const formattedPrice = Number(
                 data.predicted_price
             ).toLocaleString("en-IN", {
                 minimumFractionDigits: 2,
                 maximumFractionDigits: 2
             });

             predictedPrice.textContent =
                 `₹${formattedPrice} per quintal`;

             priceResultDate.textContent =
                 `Prediction date: ${data.date}`;

             priceResult.hidden = false;
             priceMessage.textContent =
                 "Coriander price prediction generated successfully.";

         } catch (error) {
             priceMessage.textContent =
                 "Could not predict price: " + error.message;
         } finally {
             priceButton.disabled = false;
         }
     });
 }
