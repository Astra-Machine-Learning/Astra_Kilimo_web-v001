// update form next previouss    

let currentSection = 0;
const sections = document.querySelectorAll('.quiz-section');
const progressBar = document.getElementById('progress-bar');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const submitBtn = document.getElementById('submit-btn');

function updateSection() {
    sections.forEach((section, index) => {
        section.classList.toggle('hidden', index !== currentSection);
    });
    prevBtn.disabled = currentSection === 0;
    nextBtn.classList.toggle('hidden', currentSection === sections.length - 1);
    submitBtn.classList.toggle('hidden', currentSection !== sections.length - 1);
    progressBar.style.width = `${((currentSection + 1) / sections.length) * 100}%`;

    // Check if all required inputs in the current section are filled
    const inputs = sections[currentSection].querySelectorAll('input, select');
    let allFilled = true;
    inputs.forEach(input => {
        if (input.type === 'radio' || input.type === 'checkbox') {
            const name = input.name;
            const checked = sections[currentSection].querySelector(`input[name="${name}"]:checked`);
            if (!checked) {
                allFilled = false;
            }
        } else if (!input.value) {
            allFilled = false;
        }
    });
    nextBtn.disabled = !allFilled;
}

document.querySelectorAll('input, select').forEach(input => {
    input.addEventListener('input', () => {
        updateSection();
    });
});

prevBtn.addEventListener('click', () => { currentSection--; updateSection(); });
nextBtn.addEventListener('click', () => { currentSection++; updateSection(); });

updateSection();
 
 
 
 
 
 // Get user's location and fill the location input field
 document.getElementById('get-location-btn').addEventListener('click', () => {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(position => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            document.getElementById('location-input').value = `Lat: ${lat}, Lon: ${lon}`;
            nextBtn.disabled = false;

            // Initialize Google Maps
            const map = new google.maps.Map(document.getElementById('map'), {
                center: { lat: lat, lng: lon },
                zoom: 15
            });

            // Add a marker at the user's location
            new google.maps.Marker({
                position: { lat: lat, lng: lon },
                map: map,
                title: 'Your Location'
            });
        }, error => {
            console.error('Error getting location:', error);
            alert('Unable to retrieve your location. Please ensure your browser has location permissions enabled and try again.');
        });
    } else {
        alert('Geolocation is not supported by this browser.');
    }
});

document.getElementById('location-input').addEventListener('input', () => {
    nextBtn.disabled = !document.getElementById('location-input').value;
});