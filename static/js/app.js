// Function to print JSON data to console in a readable format
function printJsonData(data) {
    console.log('=== Voronoi Diagram Data ===');
    console.log('Points:', data.points);
    console.log('Vertices:', data.vertices);
    console.log('Regions:', data.regions);
    console.log('Ridges:', data.ridges);
    console.log('===========================');
}

// Function to generate the diagram
async function generateDiagram() {
    const diagram = document.getElementById('diagram');
    const points = document.getElementById('points').value;
    const showNumbers = document.getElementById('showNumbers').checked;
    const showPoints = document.getElementById('showPoints').checked;
    
    try {
        const response = await fetch(`/generate?points=${points}&show_numbers=${showNumbers}&show_points=${showPoints}`);
        const data = await response.json();
        diagram.src = data.diagram;
        // Print the JSON data to console
        const json_data = JSON.parse(data.json_str);
        console.log(json_data)
        printJsonData(json_data);
        // Show the image after first load`
        diagram.style.display = 'block';
        drawSquareScatterPlot(json_data.points);
    } catch (error) {
        console.error('Error generating diagram:', error);
        // Still show the image even if there's an error
        diagram.style.display = 'block';
    }
}

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Generate diagram on page load
    generateDiagram();
    
    // Add event listener for form submission
    document.getElementById('voronoiForm').addEventListener('submit', function(e) {
        e.preventDefault();
        generateDiagram();
    });
});

// Function to draw a square scatter plot
function drawSquareScatterPlot(points) {
    const canvas = document.getElementById('myPlot');
    const ctx = canvas.getContext("2d");
    
    // Set canvas size
    const size = 500; // Fixed size for the canvas
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    canvas.width = size * window.devicePixelRatio;
    canvas.height = size * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw the border
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, size, size);
    
    // Destroy previous chart instance if it exists
    if (window.myChart) {
        window.myChart.destroy();
    }
    
    // Create a new chart instance
    window.myChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: '',
                data: points,
                pointBackgroundColor: 'black',
                pointRadius: 4,
                borderWidth: 0
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            layout: {
                padding: 0
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: 0,
                    max: 1,
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        display: false
                    }
                },
                y: {
                    type: 'linear',
                    min: 0,
                    max: 1,
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        display: false
                    }
                }
            }
        }
    });
}