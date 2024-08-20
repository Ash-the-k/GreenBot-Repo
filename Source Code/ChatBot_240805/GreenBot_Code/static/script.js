document.addEventListener('DOMContentLoaded', (event) => {
    const footerImageContainer = document.getElementById('footer-image-container');
    let lastScrollTop = 0;
    let debounce;
    const scrollThreshold = 50;  // Adjust this value as needed

    window.addEventListener('scroll', () => {
        clearTimeout(debounce);
        debounce = setTimeout(() => {
            const currentScrollTop = window.pageYOffset || document.documentElement.scrollTop;

            if (currentScrollTop > lastScrollTop + scrollThreshold) {
                // User is scrolling down
                footerImageContainer.style.display = 'block';
            } else if (currentScrollTop < lastScrollTop - scrollThreshold) {
                // User is scrolling up
                footerImageContainer.style.display = 'none';
            }

            lastScrollTop = currentScrollTop <= 0 ? 0 : currentScrollTop; // For Mobile or negative scrolling
        }, 100);
    });

    // Initialize the state
    footerImageContainer.style.display = 'none';
});
