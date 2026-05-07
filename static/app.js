let chart = null;

let latestForecast = [];


// =========================================
// FORMAT LARGE NUMBERS
// =========================================

function formatNumber(value) {

    value = Number(value);

    if (value >= 1000000000) {

        return (
            (value / 1000000000)
            .toFixed(2) +
            " Billion"
        );
    }

    else if (value >= 1000000) {

        return (
            (value / 1000000)
            .toFixed(2) +
            " Million"
        );
    }

    else if (value >= 1000) {

        return (
            (value / 1000)
            .toFixed(2) +
            " Thousand"
        );
    }

    return value.toString();
}


// =========================================
// LOAD FORECAST
// =========================================

async function loadForecast() {

    try {

        // SHOW LOADER

        document.getElementById(
            "loader"
        ).style.display = "flex";


        const state =
            document.getElementById(
                "stateSelect"
            ).value;

        const response = await fetch(
            `/forecast?state=${state}`
        );

        if (!response.ok) {

            throw new Error(
                "API Error"
            );
        }

        const data =
            await response.json();

        latestForecast =
            data["8_week_forecast"];

        // KPI

        document.getElementById(
            "bestModel"
        ).innerText =
            data.best_model;

        document.getElementById(
            "mae"
        ).innerText =
            formatNumber(data.mae);

        document.getElementById(
            "rmse"
        ).innerText =
            formatNumber(data.rmse);


        // TIMESTAMP

        const now = new Date();

        document.getElementById(
            "timestamp"
        ).innerText =
            `Forecast generated at: ${now.toLocaleString()}`;


        // TABLE

        const tableBody =
            document.getElementById(
                "forecastTableBody"
            );

        tableBody.innerHTML = "";

        latestForecast.forEach(
            (value, index) => {

                tableBody.innerHTML += `

                    <tr>

                        <td>
                            Week ${index + 1}
                        </td>

                        <td>
                            ${formatNumber(value)}
                        </td>

                    </tr>

                `;
            }
        );


        // CHART

        const ctx =
            document.getElementById(
                "forecastChart"
            ).getContext("2d");

        if (chart) {

            chart.destroy();
        }

        chart = new Chart(ctx, {

            type: "line",

            data: {

                labels: [

                    "Week 1",
                    "Week 2",
                    "Week 3",
                    "Week 4",
                    "Week 5",
                    "Week 6",
                    "Week 7",
                    "Week 8"
                ],

                datasets: [

                    {

                        label:
                            `${state} Sales Forecast`,

                        data:
                            latestForecast,

                        borderColor:
                            "#4F8CFF",

                        backgroundColor:
                            "rgba(79,140,255,0.2)",

                        borderWidth: 3,

                        pointRadius: 5,

                        tension: 0.4,

                        fill: true
                    }
                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false
            }
        });


        // HIDE LOADER

        document.getElementById(
            "loader"
        ).style.display = "none";
    }

    catch (error) {

        console.error(error);

        document.getElementById(
            "loader"
        ).style.display = "none";

        alert(
            "Error loading forecast."
        );
    }
}


// =========================================
// DOWNLOAD CSV
// =========================================

function downloadCSV() {

    if (latestForecast.length === 0) {

        alert(
            "Generate forecast first."
        );

        return;
    }

    let csvContent =
        "Week,Predicted Sales\n";

    latestForecast.forEach(
        (value, index) => {

            csvContent +=
                `Week ${index + 1},${value}\n`;
        }
    );

    const blob = new Blob(
        [csvContent],
        { type: "text/csv" }
    );

    const url =
        window.URL.createObjectURL(blob);

    const a =
        document.createElement("a");

    a.href = url;

    a.download =
        "sales_forecast.csv";

    a.click();

    window.URL.revokeObjectURL(url);
}


// =========================================
// TOGGLE THEME
// =========================================

function toggleTheme() {

    document.body.classList.toggle(
        "light-mode"
    );
}