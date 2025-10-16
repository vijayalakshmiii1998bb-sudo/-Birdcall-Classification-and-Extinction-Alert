// bird/static/bird/app.js
document.addEventListener('DOMContentLoaded', () => {
  if (window.gsap) {
    gsap.from('h1, h2', {y: 18, opacity: 0, duration: 0.8, stagger: 0.08, ease: "power2"});
    gsap.from('.card', {y: 20, opacity: 0, duration: 0.6, stagger: 0.05, ease: "power2.out"});
    gsap.from('.btn-primary, .btn-secondary', {scale: 0.9, opacity: 0, duration: .4, stagger: .05});
  }
});
